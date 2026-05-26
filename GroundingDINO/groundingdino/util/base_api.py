from typing import Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.misc import clean_state_dict, clean_state_dict_test
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import get_phrases_from_posmap


def preprocess_caption(caption: str) -> str:
    result = caption.lower().strip()
    if result.endswith("."):
        return result
    return result + "."


def _torch_load_checkpoint(model_checkpoint_path: str):
    try:
        checkpoint = torch.load(model_checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    return checkpoint


def infer_config_overrides_from_checkpoint(model_checkpoint_path: str) -> Dict[str, bool]:
    checkpoint = _torch_load_checkpoint(model_checkpoint_path)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    keys = list(state_dict.keys()) if isinstance(state_dict, dict) else []
    use_ddg = any(k.startswith("transformer.dual_density_regressor.") for k in keys)
    use_sdpr = any(k.startswith("transformer.dual_density_regressor.sdpr.") for k in keys)
    if use_sdpr:
        use_ddg = True
    return {"use_ddg": use_ddg, "use_sdpr": use_sdpr}


def load_model(
    model_config_path: str,
    model_checkpoint_path: str,
    device: str = "cuda",
    mode: str = 'train',
    config_overrides: Optional[Dict[str, object]] = None,
):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    if config_overrides is not None:
        for key, value in config_overrides.items():
            setattr(args, key, value)
    model = build_model(args)
    checkpoint = _torch_load_checkpoint(model_checkpoint_path)
    if mode == 'train':
        model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        model.eval()
        return model
    else:
        model.load_state_dict(clean_state_dict_test(checkpoint["model"]), strict=False)
        model.eval()
        return model
        

def load_image(image_path: str) -> Tuple[np.array, torch.Tensor]:
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_source = Image.open(image_path).convert("RGB")
    image = np.asarray(image_source)
    image_transformed, _ = transform(image_source, None)
    return image, image_transformed


def threshold(
        outputs,
        captions: str,
        tokenizer,
        text_threshold: float,
        threshold1 = 0.25,
        threshold2 = 0.35): 

    bs = outputs["pred_logits"].shape[0]

    ret = []
    for b in range(bs):
        prediction_logits = outputs["pred_logits"].cpu().sigmoid()[b]  
        prediction_boxes = outputs["pred_boxes"].cpu()[b]  

        tokenized = tokenizer(captions[b])
        input_ids = tokenized['input_ids']
        end_idx = np.where(np.array(input_ids)==1012)[0][-1]
        
        # find mask index where all the valid tokens are above the threshold
        threshold1 = threshold1
        threshold2 = threshold2
        # for global context
        mask1 = prediction_logits[:, 0].gt(threshold1)
        # for local context 找到对于所有文本的响应都高于目标的。
        mask2 = prediction_logits[:, 1:end_idx].gt(threshold2).all(dim=1) 
        mask = mask1 & mask2

        logits = prediction_logits[mask]  
        boxes = prediction_boxes[mask]  


        phrases = [ 
            get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer).replace('.', '')
            for logit 
            in logits
        ]
        ret.append((boxes, logits.max(dim=1)[0], phrases))

    return ret

def threshold_box(
        outputs, threshold=0.3): 

    bs = outputs["pred_logits"].shape[0]

    ret = []
    for b in range(bs):
        prediction_logits = outputs["pred_logits"].cpu().sigmoid()[b]  
        prediction_boxes = outputs["pred_boxes"].cpu()[b]  

        # find mask index where all the valid tokens are above the threshold
        threshold = threshold
        # for global context
        # for local context 找到对于所有文本的响应都高于目标的。
        mask = prediction_logits[:, :].gt(threshold).any(dim=1) 

        logits = prediction_logits[mask]  
        boxes = prediction_boxes[mask]  

        ret.append((boxes, logits.max(dim=1)[0]))

    return ret
