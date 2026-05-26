#!/usr/bin/env python3
"""
Precompute Stable Diffusion cross-attention maps for REC-8K dataset.
This script extracts cross-attention maps from SD UNet as auxiliary supervision signals.

Usage:
    python precompute_sd_attention.py \
        --data_root /path/to/rec-8k \
        --anno_file /path/to/annotations.json \
        --output_dir /path/to/output
"""

import os
import json
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from diffusers import StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel
from diffusers.models.attention_processor import Attention
from transformers import CLIPTextModel, CLIPTokenizer


class CrossAttentionHook:
    """Hook to capture cross-attention maps from UNet."""
    
    def __init__(self):
        self.attention_maps: List[torch.Tensor] = []
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
    
    def clear(self):
        """Clear stored attention maps."""
        self.attention_maps.clear()
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def _hook_fn(self, module: torch.nn.Module, args, kwargs, output):
        """Hook function to capture attention output."""
        # In diffusers 0.37.0, Attention modules have hidden_states as output
        # We need to capture the attention weights during forward pass
        pass
    
    def register_hooks(self, unet: UNet2DConditionModel):
        """Register forward hooks on cross-attention layers."""
        self.remove_hooks()
        
        for name, module in unet.named_modules():
            # In diffusers, cross-attention is typically named with 'attn2'
            if isinstance(module, Attention) and 'attn2' in name:
                # We'll use a custom attention processor to capture attention maps
                pass
        
        return self


class AttentionMapProcessor:
    """Custom attention processor that captures attention maps."""
    
    def __init__(self, original_processor, attention_store: List):
        self.original_processor = original_processor
        self.attention_store = attention_store
    
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        # Only capture cross-attention (when encoder_hidden_states is provided)
        if encoder_hidden_states is not None:
            batch_size, sequence_length, _ = hidden_states.shape
            
            # Get query, key from attention module
            query = attn.to_q(hidden_states)
            key = attn.to_k(encoder_hidden_states)
            
            # Reshape for multi-head attention
            inner_dim = key.shape[-1]
            head_dim = inner_dim // attn.heads
            
            query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            
            # Compute attention weights
            scale = head_dim ** -0.5
            attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scale
            attn_weights = attn_weights.softmax(dim=-1)
            
            # Store attention maps: (batch, heads, seq_len, text_len)
            self.attention_store.append(attn_weights.detach().cpu())
        
        # Call original processor for actual computation
        return self.original_processor(
            attn,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            temb=temb,
            *args,
            **kwargs,
        )


class SDAttentionExtractor:
    """Extract cross-attention maps from Stable Diffusion."""
    
    def __init__(
        self,
        sd_model: str = "runwayml/stable-diffusion-v1-5",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        self.device = device
        self.dtype = dtype
        
        print(f"Loading Stable Diffusion model: {sd_model}")
        
        # Load pipeline
        self.pipe = StableDiffusionPipeline.from_pretrained(
            sd_model,
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )
        self.pipe = self.pipe.to(device)
        
        # Get components
        self.vae: AutoencoderKL = self.pipe.vae
        self.unet: UNet2DConditionModel = self.pipe.unet
        self.text_encoder: CLIPTextModel = self.pipe.text_encoder
        self.tokenizer: CLIPTokenizer = self.pipe.tokenizer
        self.scheduler = self.pipe.scheduler
        
        # Set to eval mode
        self.vae.eval()
        self.unet.eval()
        self.text_encoder.eval()
        
        # Attention map storage
        self.attention_maps: List[torch.Tensor] = []
        self.original_processors: Dict = {}
        
        print("Model loaded successfully!")
    
    def _register_attention_hooks(self):
        """Register custom attention processors to capture attention maps."""
        self.attention_maps.clear()
        self.original_processors.clear()
        
        for name, module in self.unet.named_modules():
            if isinstance(module, Attention) and 'attn2' in name:
                # Store original processor
                self.original_processors[name] = module.processor
                # Set custom processor
                module.processor = AttentionMapProcessor(
                    module.processor, 
                    self.attention_maps
                )
    
    def _restore_attention_processors(self):
        """Restore original attention processors."""
        for name, module in self.unet.named_modules():
            if name in self.original_processors:
                module.processor = self.original_processors[name]
        self.original_processors.clear()
    
    @torch.no_grad()
    def encode_image(self, image: Image.Image, target_size: int = 512) -> torch.Tensor:
        """Encode image to latent space using VAE."""
        # Resize and normalize image
        image = image.convert("RGB")
        image = image.resize((target_size, target_size), Image.LANCZOS)
        
        # Convert to tensor: (H, W, C) -> (C, H, W)
        image_tensor = torch.tensor(list(image.getdata()), dtype=torch.float32)
        image_tensor = image_tensor.view(target_size, target_size, 3)
        image_tensor = image_tensor.permute(2, 0, 1)
        
        # Normalize to [-1, 1]
        image_tensor = (image_tensor / 127.5) - 1.0
        image_tensor = image_tensor.unsqueeze(0).to(self.device, dtype=self.dtype)
        
        # Encode
        latent = self.vae.encode(image_tensor).latent_dist.sample()
        latent = latent * self.vae.config.scaling_factor
        
        return latent
    
    @torch.no_grad()
    def encode_text(self, text: str) -> torch.Tensor:
        """Encode text using CLIP text encoder."""
        # Tokenize
        text_inputs = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.device)
        
        # Encode
        text_embeddings = self.text_encoder(text_input_ids)[0]
        
        return text_embeddings
    
    @torch.no_grad()
    def extract_attention(
        self,
        image: Image.Image,
        text: str,
        timestep: int = 50,
        resolution: int = 64,
    ) -> torch.Tensor:
        """
        Extract cross-attention map for image-text pair.
        
        Args:
            image: Input image
            text: Referring expression text
            timestep: Noise timestep (0-1000)
            resolution: Output attention map resolution
        
        Returns:
            Attention map tensor of shape (1, resolution, resolution)
        """
        # Encode image to latent
        latent = self.encode_image(image)
        
        # Encode text
        text_embeddings = self.encode_text(text)
        
        # Add noise at specified timestep
        noise = torch.randn_like(latent)
        timesteps = torch.tensor([timestep], device=self.device, dtype=torch.long)
        
        # Set scheduler timesteps
        self.scheduler.set_timesteps(1000)
        
        # Add noise using scheduler
        noisy_latent = self.scheduler.add_noise(latent, noise, timesteps)
        
        # Register hooks
        self._register_attention_hooks()
        
        try:
            # Single UNet forward pass
            _ = self.unet(
                noisy_latent,
                timesteps,
                encoder_hidden_states=text_embeddings,
                return_dict=False,
            )
        finally:
            # Restore original processors
            self._restore_attention_processors()
        
        # Aggregate attention maps
        aggregated_attn = self._aggregate_attention_maps(resolution)
        
        return aggregated_attn
    
    def _aggregate_attention_maps(self, target_resolution: int = 64) -> torch.Tensor:
        """
        Aggregate multi-layer, multi-head attention maps to unified resolution.
        
        Args:
            target_resolution: Target spatial resolution
        
        Returns:
            Aggregated attention map of shape (1, H, W)
        """
        if not self.attention_maps:
            raise ValueError("No attention maps captured!")
        
        aggregated = None
        count = 0
        
        for attn_map in self.attention_maps:
            # attn_map shape: (batch, heads, seq_len, text_len)
            batch, heads, seq_len, text_len = attn_map.shape
            
            # Compute spatial resolution (seq_len = h * w for image tokens)
            spatial_size = int(seq_len ** 0.5)
            
            if spatial_size * spatial_size != seq_len:
                # Skip non-square attention maps
                continue
            
            # Average over heads: (batch, seq_len, text_len)
            attn_map = attn_map.mean(dim=1)
            
            # Sum over text tokens (excluding special tokens like [CLS], [SEP])
            # We use tokens 1:-1 to exclude start and end tokens
            # But for simplicity, we average all text tokens
            attn_map = attn_map.mean(dim=-1)  # (batch, seq_len)
            
            # Reshape to spatial: (batch, h, w)
            attn_map = attn_map.view(batch, spatial_size, spatial_size)
            
            # Resize to target resolution
            attn_map = F.interpolate(
                attn_map.unsqueeze(1).float(),
                size=(target_resolution, target_resolution),
                mode='bilinear',
                align_corners=False,
            ).squeeze(1)
            
            if aggregated is None:
                aggregated = attn_map
            else:
                aggregated = aggregated + attn_map
            count += 1
        
        if aggregated is None or count == 0:
            raise ValueError("Failed to aggregate attention maps!")
        
        # Average over layers
        aggregated = aggregated / count
        
        # Normalize to [0, 1]
        aggregated = aggregated - aggregated.min()
        if aggregated.max() > 0:
            aggregated = aggregated / aggregated.max()
        
        # Shape: (1, H, W)
        return aggregated
    
    def clear_attention_maps(self):
        """Clear stored attention maps."""
        self.attention_maps.clear()


def load_annotations(anno_file: str) -> Dict:
    """Load annotations from JSON file."""
    with open(anno_file, 'r') as f:
        return json.load(f)


def get_output_filename(image_id: str, expr_index: int) -> str:
    """Generate output filename for attention map."""
    # Remove extension from image_id
    base_name = os.path.splitext(image_id)[0]
    return f"{base_name}_{expr_index}.pt"


def process_dataset(
    extractor: SDAttentionExtractor,
    data_root: str,
    anno_file: str,
    output_dir: str,
    timestep: int = 50,
    resolution: int = 64,
    batch_size: int = 1,  # Currently only batch_size=1 is supported
):
    """Process entire dataset and save attention maps."""
    # Load annotations
    print(f"Loading annotations from: {anno_file}")
    annotations = load_annotations(anno_file)
    print(f"Found {len(annotations)} images")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Count total expressions for progress bar
    total_expressions = sum(len(exprs) for exprs in annotations.values())
    print(f"Total expressions to process: {total_expressions}")
    
    # Process images
    processed = 0
    skipped = 0
    errors = 0
    
    with tqdm(
        total=total_expressions,
        desc="SD attention",
        unit="expr",
        dynamic_ncols=True,
        file=sys.stdout,
    ) as pbar:
        for image_id, expressions in annotations.items():
            image_path = os.path.join(data_root, image_id)
            
            if not os.path.exists(image_path):
                tqdm.write(f"Warning: Image not found: {image_path}")
                pbar.update(len(expressions))
                errors += len(expressions)
                pbar.set_postfix(
                    image=image_id,
                    processed=processed,
                    skipped=skipped,
                    errors=errors,
                )
                continue
            
            # Load image once per image
            try:
                image = Image.open(image_path)
            except Exception as e:
                tqdm.write(f"Error loading image {image_path}: {e}")
                pbar.update(len(expressions))
                errors += len(expressions)
                pbar.set_postfix(
                    image=image_id,
                    processed=processed,
                    skipped=skipped,
                    errors=errors,
                )
                continue
            
            # Process each expression
            for expr_index, (expr_text, expr_data) in enumerate(expressions.items()):
                output_filename = get_output_filename(image_id, expr_index)
                output_path = os.path.join(output_dir, output_filename)
                
                # Skip if already processed (resume capability)
                if os.path.exists(output_path):
                    skipped += 1
                    pbar.update(1)
                    pbar.set_postfix(
                        image=image_id,
                        processed=processed,
                        skipped=skipped,
                        errors=errors,
                    )
                    continue
                
                try:
                    # Extract attention map
                    attn_map = extractor.extract_attention(
                        image=image,
                        text=expr_text,
                        timestep=timestep,
                        resolution=resolution,
                    )
                    
                    # Save attention map
                    # Shape: (1, H, W), dtype: float32, range: [0, 1]
                    torch.save(attn_map.float(), output_path)
                    processed += 1
                    
                except Exception as e:
                    tqdm.write(f"Error processing {image_id}/{expr_text}: {e}")
                    errors += 1
                
                # Clear attention maps for next iteration
                extractor.clear_attention_maps()
                pbar.update(1)
                pbar.set_postfix(
                    image=image_id,
                    processed=processed,
                    skipped=skipped,
                    errors=errors,
                )
            
            # Close image to free memory
            image.close()
    
    print(f"\nProcessing complete!")
    print(f"  Processed: {processed}")
    print(f"  Skipped (already exists): {skipped}")
    print(f"  Errors: {errors}")


def main():
    parser = argparse.ArgumentParser(
        description="Precompute SD cross-attention maps for REC-8K dataset"
    )
    parser.add_argument(
        "--sd_model",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Stable Diffusion model path or HuggingFace model ID",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/root/autodl-tmp/data/rec-8k/",
        help="REC-8K image directory",
    )
    parser.add_argument(
        "--anno_file",
        type=str,
        default="/root/autodl-tmp/data/rec-8k/annotations.json",
        help="Annotations JSON file path",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/root/autodl-tmp/data/rec-8k-sd-attn/",
        help="Output directory for attention maps",
    )
    parser.add_argument(
        "--timestep",
        type=int,
        default=50,
        help="Noise timestep for attention extraction (default: 50)",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=64,
        help="Attention map resolution (default: 64)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size (default: 1, currently only 1 is supported)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (default: cuda)",
    )
    parser.add_argument(
        "--fp32",
        action="store_true",
        help="Use FP32 instead of FP16",
    )
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.data_root):
        raise FileNotFoundError(f"Data root not found: {args.data_root}")
    if not os.path.exists(args.anno_file):
        raise FileNotFoundError(f"Annotation file not found: {args.anno_file}")
    
    # Set dtype
    dtype = torch.float32 if args.fp32 else torch.float16
    
    # Initialize extractor
    extractor = SDAttentionExtractor(
        sd_model=args.sd_model,
        device=args.device,
        dtype=dtype,
    )
    
    # Process dataset
    process_dataset(
        extractor=extractor,
        data_root=args.data_root,
        anno_file=args.anno_file,
        output_dir=args.output_dir,
        timestep=args.timestep,
        resolution=args.resolution,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
