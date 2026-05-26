import json
import os
from torch.utils.data import DataLoader

from .fsc147 import FSC147Dataset, MainTransform, get_query_transforms, batch_collate_fn


def _resolve_fsc147_root():
    candidates = []
    env_root = os.environ.get("FSC147_ROOT", "").strip()
    if env_root:
        candidates.append(env_root)
    candidates.extend([
        "/root/autodl-tmp/data/fsc147",
        "/root/autodl-tmp/data/FSC147_384_V2",
        "./datasets/fsc147",
    ])

    for root in candidates:
        if not root:
            continue
        anno = os.path.join(root, "annotation_FSC147_384.json")
        img_dir = os.path.join(root, "images_384_VarV2")
        den_dir = os.path.join(root, "gt_density_map_adaptive_384_VarV2")
        if os.path.exists(anno) and os.path.isdir(img_dir) and os.path.isdir(den_dir):
            return root
    raise FileNotFoundError(
        "FSC147 dataset root not found. Please set FSC147_ROOT or place data under "
        "/root/autodl-tmp/data/FSC147_384_V2."
    )


def _resolve_split_list(data_root, split):
    txt_path = os.path.join(data_root, f"{split}.txt")
    if os.path.exists(txt_path):
        return txt_path

    split_json = os.path.join(data_root, "Train_Test_Val_FSC_147.json")
    if not os.path.exists(split_json):
        raise FileNotFoundError(f"Missing split file: {txt_path} and {split_json}")

    with open(split_json, "r") as f:
        split_data = json.load(f)
    if split not in split_data:
        raise KeyError(f"Split '{split}' not found in {split_json}")
    return split_data[split]

def get_fsc_loader(split, batch_size, args=None):
    main_transform = MainTransform()
    query_transform = get_query_transforms(True, (128,128))
    data_root = _resolve_fsc147_root()
    data_list = _resolve_split_list(data_root, split)
    
    if not args:
        dataset = FSC147Dataset(data_dir=data_root,
                            data_list=data_list,
                            scaling=1.0,
                            box_number=3,
                            scale_number=1,
                            main_transform=main_transform,
                            query_transform=query_transform,
                            split=split)
    else:
        dataset = FSC147Dataset(data_dir=data_root,
                    data_list=data_list,
                    scaling=1.0,
                    box_number=3,
                    scale_number=1,
                    main_transform=main_transform,
                    query_transform=query_transform,
                    split=split,
                    horizon_flip=args.horizon_flip,
                    horizon_flip_prob=args.horizon_flip_prob,
                    vertical_flip=args.vertical_flip,
                    vertical_flip_prob=args.vertical_flip_prob)
    shuffle = True if split == 'train' else False
    split_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=batch_collate_fn)
    return split_loader
    


