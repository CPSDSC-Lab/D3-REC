import os
import torch
import numpy as np
import copy
import re
import cv2
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from groundingdino.util.base_api import load_model, threshold, threshold_box
import os
import numpy as np
from datetime import datetime
import math

from utils.processor import DataProcessor
from utils.criterion import SetCriterion, L2Loss, SetRegContrastiveCriterion
from utils.criterion_box import SetCriterionBox, SetCriterionFSC147
from utils.image_loader import get_loader
from utils.image_loader_fsc147 import get_fsc_loader
from utils.support_memory import SupportMemory, extract_query_features, save_support_memory
from tqdm import tqdm
import torchvision.transforms.functional as TF

device = 'cuda' if torch.cuda.is_available() else 'cpu'
criterion_localization = SetCriterion()
criterion_counting = L2Loss()

COLOR_HSV_RANGES = {
    "red": [((0, 70, 50), (10, 255, 255)), ((170, 70, 50), (180, 255, 255))],
    "green": [((35, 50, 40), (90, 255, 255))],
    "blue": [((90, 50, 40), (140, 255, 255))],
    "yellow": [((18, 70, 70), (35, 255, 255))],
    "black": [((0, 0, 0), (180, 255, 60))],
    "white": [((0, 0, 180), (180, 60, 255))],
}


def _extract_color_word(caption):
    low = caption.lower()
    for c in COLOR_HSV_RANGES.keys():
        if re.search(rf"\b{c}\b", low):
            return c
    return None


def _to_hsv_mask(image_tensor, color_word):
    if color_word is None:
        return None
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    img = image_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    img = (img * std + mean).clip(0.0, 1.0)
    img_uint8 = (img * 255.0).astype(np.uint8)
    hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in COLOR_HSV_RANGES[color_word]:
        m = cv2.inRange(hsv, np.array(lo, dtype=np.uint8), np.array(hi, dtype=np.uint8))
        mask = np.maximum(mask, m)
    return mask > 0


def _box_color_ratio(mask, box_cxcywh):
    if mask is None:
        return 1.0
    h, w = mask.shape
    cx, cy, bw, bh = box_cxcywh
    x1 = int(max(0, min(w - 1, (cx - bw / 2.0) * w)))
    x2 = int(max(0, min(w, (cx + bw / 2.0) * w)))
    y1 = int(max(0, min(h - 1, (cy - bh / 2.0) * h)))
    y2 = int(max(0, min(h, (cy + bh / 2.0) * h)))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    region = mask[y1:y2, x1:x2]
    if region.size == 0:
        return 0.0
    return float(region.mean())

def threshold_dynamic(
        outputs,
        pred_num): 
    bs = outputs["pred_logits"].shape[0]
    ret = []
    b = 0
    for b in range(bs):
        prediction_logits = outputs["pred_logits"].cpu().sigmoid()[b]  
        prediction_boxes = outputs["pred_boxes"].cpu()[b]  
        
        logits = prediction_logits[:,0]
        topk = pred_num[b]
        values, indices = torch.topk(logits.squeeze(), topk)
        ret.append((prediction_boxes[indices], logits[indices]))

    return ret

def threshold_dynamic_context(
        outputs,
        captions: str,
        images,
        tokenizer,
        pred_num,
        threshold1_list=None,
        threshold2_list=None,
        k_bias_list=None,
        k_min=1,
        k_max=900,
        color_prefilter=False,
        color_prefilter_min_ratio=0.08): 
    bs = outputs["pred_logits"].shape[0]
    ret = []
    b = 0
    for b in range(bs):
        prediction_logits = outputs["pred_logits"].cpu().sigmoid()[b]  
        prediction_boxes = outputs["pred_boxes"].cpu()[b]  
        
        tokenized = tokenizer(captions[b])
        input_ids = tokenized['input_ids']
        end_idx = np.where(np.array(input_ids)==1012)[0][-1]
        
        prediction_logits_context = torch.mean(prediction_logits[:, 1:end_idx],dim=1)
        logits = prediction_logits_context
        candidate_boxes = prediction_boxes
        candidate_logits = logits
        if threshold1_list is not None and threshold2_list is not None:
            t1 = float(threshold1_list[b])
            t2 = float(threshold2_list[b])
            mask_global = prediction_logits[:, 0].gt(t1)
            if end_idx > 1:
                mask_local = prediction_logits[:, 1:end_idx].gt(t2).all(dim=1)
                mask = mask_global & mask_local
            else:
                mask = mask_global
            if mask.any():
                candidate_boxes = prediction_boxes[mask]
                candidate_logits = logits[mask]
        if color_prefilter:
            color_word = _extract_color_word(captions[b])
            if color_word is not None and candidate_boxes.numel() > 0:
                color_mask = _to_hsv_mask(images[b], color_word)
                keep = []
                for i in range(candidate_boxes.shape[0]):
                    keep.append(_box_color_ratio(color_mask, candidate_boxes[i].tolist()) >= float(color_prefilter_min_ratio))
                keep = torch.tensor(keep, dtype=torch.bool)
                if keep.any():
                    candidate_boxes = candidate_boxes[keep]
                    candidate_logits = candidate_logits[keep]
        topk = int(pred_num[b])
        if k_bias_list is not None:
            topk = int(round(topk + float(k_bias_list[b])))
        topk = max(int(k_min), min(int(k_max), topk))
        topk = min(topk, int(candidate_logits.numel()))
        if topk <= 0:
            ret.append((candidate_boxes[:0], candidate_logits[:0]))
            continue
        values, indices = torch.topk(candidate_logits.squeeze(), topk)
        ret.append((candidate_boxes[indices], candidate_logits[indices]))

    return ret

def eval(split, model, loaders, annotations, args):
    print(f"Evaluation on {split} set")
    model.eval()
    loader = loaders[split]
    support_memory = None
    if getattr(args, "use_support", False):
        support_memory = SupportMemory.load(getattr(args, "support_path", None), device=device)
        if support_memory is not None:
            print(f"Loaded support memory: {support_memory.size}")
        else:
            print("Support memory unavailable, fallback to CAD-GD† dynamic top-k")
    support_records = []

    eval_mae = 0
    eval_rmse = 0

    val_mae = 0
    val_rmse = 0

    eval_tp = 0
    eval_fp = 0
    eval_fn = 0
    
    counter = 0
    counter_for_image = 0
    eval_size = len(loader.dataset)

    total_num = 0

    for batch in tqdm(loader): # tensor, list, list, list
        if len(batch) == 6:
            images, captions, shapes, img_caps, density_maps, sd_attn_maps = batch
        else:
            images, captions, shapes, img_caps, density_maps = batch
            sd_attn_maps = None
        mask_bi = [i for i, img_cap_list in enumerate(img_caps) for _ in img_cap_list] # index for each img,cap pair in the batch
        anno_b = [annotations[img_cap] for img_cap_list in img_caps for img_cap in img_cap_list] 
        img_caps = [img_cap for img_cap_list in img_caps for img_cap in img_cap_list]
        shapes = [shapes[i] for i, caption_list in enumerate(captions) for _ in caption_list]
        images = torch.stack([images[i] for i, caption_list in enumerate(captions) for _ in caption_list], dim=0)
        captions = [caption for caption_list in captions for caption in caption_list] # flatten list of list
        images = images.to(device)
        sd_attn_map_tensor = None
        if sd_attn_maps is not None and len(sd_attn_maps) > 0 and sd_attn_maps[0] is not None:
            valid_maps = [m for m in sd_attn_maps if m is not None]
            if len(valid_maps) > 0:
                sd_attn_map_tensor = torch.stack(valid_maps, dim=0).to(device)
        with torch.no_grad():
            outputs = model(images, captions=captions, image_name=img_caps[0][0], sd_attn_map=sd_attn_map_tensor)
        if sd_attn_map_tensor is not None:
            outputs['sd_attn_map'] = sd_attn_map_tensor
        ## density output
        pred_density_map = outputs['density'].squeeze(1)
        density_maps = density_maps.to(pred_density_map.device)
        ### mae 
        total_num += density_maps.shape[0]
        counter_for_image += 1

        ## localization output
        outputs["pred_points"] = outputs["pred_boxes"][:, :, :2] 

        # prepare targets
        emb_size = outputs["pred_logits"].shape[2]
        targets = prepare_targets(model, anno_b, captions, shapes, emb_size)
        # pred_num 
        pred_num = torch.sum(pred_density_map, dim=[1,2]) / args.scale
        gt_num_density = torch.sum(density_maps, dim = [1,2])
        for i, p in enumerate(targets):
            # gt_num = len(p['points'])
            cnt_err = torch.abs(pred_num[i] - gt_num_density[i])
            val_mae += cnt_err
            val_rmse += cnt_err ** 2
        
        pred_num_round = [round(i.item()) for i in pred_num]

        threshold1_list = None
        threshold2_list = None
        k_bias_list = None
        if support_memory is not None:
            q_img_feat, q_txt_feat = extract_query_features(outputs)
            prior = support_memory.retrieve(
                q_img_feat,
                q_txt_feat,
                topk=getattr(args, "support_topk", 8),
                alpha=getattr(args, "support_alpha", 0.5),
                tau=getattr(args, "support_tau", 0.07),
            )
            if prior is not None:
                base_t1 = float(getattr(args, "threshold1", 0.25))
                base_t2 = float(getattr(args, "threshold2", 0.35))
                gamma = float(getattr(args, "support_gamma", 0.6))
                threshold1_list = (1 - gamma) * base_t1 + gamma * prior["t1_prior"].cpu()
                threshold2_list = (1 - gamma) * base_t2 + gamma * prior["t2_prior"].cpu()
                threshold1_list = threshold1_list.clamp(
                    float(getattr(args, "support_t1_min", 0.1)),
                    float(getattr(args, "support_t1_max", 0.6)),
                )
                threshold2_list = threshold2_list.clamp(
                    float(getattr(args, "support_t2_min", 0.1)),
                    float(getattr(args, "support_t2_max", 0.7)),
                )
                k_bias_list = prior["k_bias_prior"].cpu().clamp(
                    -float(getattr(args, "support_k_bias_clip", 20)),
                    float(getattr(args, "support_k_bias_clip", 20)),
                )

        results = threshold_dynamic_context(
            outputs,
            captions,
            images,
            model.tokenizer,
            pred_num_round,
            threshold1_list=threshold1_list,
            threshold2_list=threshold2_list,
            k_bias_list=k_bias_list,
            k_min=getattr(args, "support_k_min", 1),
            k_max=getattr(args, "support_k_max", 900),
            color_prefilter=getattr(args, "color_prefilter", False),
            color_prefilter_min_ratio=getattr(args, "color_prefilter_min_ratio", 0.08),
        )

        if getattr(args, "support_collect_path", ""):
            q_img_feat, q_txt_feat = extract_query_features(outputs)
            pred_logits = outputs["pred_logits"].detach().sigmoid().cpu()
            for b in range(len(pred_num_round)):
                tokenized = model.tokenizer(captions[b])
                input_ids = tokenized['input_ids']
                end_idx = np.where(np.array(input_ids)==1012)[0][-1]
                score_context = torch.mean(pred_logits[b][:, 1:end_idx], dim=1)
                conf_mean = float(score_context.mean().item()) if score_context.numel() > 0 else 0.0
                t1 = float(getattr(args, "threshold1", 0.25))
                t2 = float(getattr(args, "threshold2", 0.35))
                conf_shift = (0.5 - conf_mean) * float(getattr(args, "support_conf_scale", 0.2))
                t1 = max(float(getattr(args, "support_t1_min", 0.1)), min(float(getattr(args, "support_t1_max", 0.6)), t1 - conf_shift))
                t2 = max(float(getattr(args, "support_t2_min", 0.1)), min(float(getattr(args, "support_t2_max", 0.7)), t2 - conf_shift))
                mask_global = pred_logits[b][:, 0].gt(t1)
                if end_idx > 1:
                    mask_local = pred_logits[b][:, 1:end_idx].gt(t2).all(dim=1)
                    base_count = int((mask_global & mask_local).sum().item())
                else:
                    base_count = int(mask_global.sum().item())
                k_bias = float(np.clip(pred_num_round[b] - base_count, -float(getattr(args, "support_k_bias_clip", 20)), float(getattr(args, "support_k_bias_clip", 20))))
                support_records.append(
                    {
                        "img_feat": q_img_feat[b].detach().cpu(),
                        "txt_feat": q_txt_feat[b].detach().cpu(),
                        "t1": t1,
                        "t2": t2,
                        "k_bias": k_bias,
                    }
                )

        # results = threshold(outputs, captions, model.tokenizer, text_threshold=0, threshold1=args.threshold1, threshold2=args.threshold2)
        for b in range(len(results)):
            boxes, logits = results[b]
            boxes = [box.tolist() for box in boxes]
            logits = logits.tolist()


            points = [[box[0], box[1]] for box in boxes]

            # calculate error
            pred_cnt = len(points)
            gt_cnt = len(targets[b]["points"])
            cnt_err = abs(pred_cnt - gt_cnt)
            eval_mae += cnt_err
            eval_rmse += cnt_err ** 2
        
            # calculate loc metric
            TP, FP, FN, precision, recall, f1 = calc_loc_metric(boxes, targets[b]["points"])
            eval_tp += TP
            eval_fp += FP
            eval_fn += FN


            counter += 1
            
            # print(f'[{split}] ep {epoch} ({counter_for_image}/{eval_size}), {img_caps[b]}, caption: {captions[b]}, actual-predicted: {gt_cnt} vs {pred_cnt}, error: {pred_cnt - gt_cnt}. Current MAE: {int(eval_mae/counter)}, RMSE: {int((eval_rmse/counter)**0.5)} | TP = {TP}, FP = {FP}, FN = {FN}, precision = {precision:.2f}, recall = {recall:.2f}, F1 = {f1:.2f}')
        
    eval_mae_regression = round(val_mae.item()/total_num,2)
    eval_rmse_regression = round(val_rmse.item()/total_num,2)**0.5

    eval_mae = eval_mae / counter
    eval_rmse = (eval_rmse / counter) ** 0.5

    eval_precision = eval_tp / (eval_tp + eval_fp) if eval_tp + eval_fp != 0 else 0.0
    eval_recall = eval_tp / (eval_tp + eval_fn) if eval_tp + eval_fn != 0 else 0.0
    eval_f1 = 2 * eval_precision * eval_recall / (eval_precision + eval_recall) if eval_precision + eval_recall != 0 else 0.0

    if getattr(args, "support_collect_path", "") and len(support_records) > 0:
        save_support_memory(args.support_collect_path, support_records)
        print(f"Saved support memory: {len(support_records)} -> {args.support_collect_path}")

    return eval_mae, eval_rmse, eval_tp, eval_fp, eval_fn, eval_precision, eval_recall, eval_f1, eval_mae_regression, eval_rmse_regression


def prepare_targets(model, anno_b, captions, shapes, emb_size):
    for anno in anno_b:
        if len(anno['points']) == 0:
            anno['points'] = [[0,0]]
    gt_points_b = [np.array(anno['points']) / np.array(shape)[::-1] for anno, shape in zip(anno_b, shapes)] # (h,w) -> (w,h)
    gt_points = [torch.from_numpy(img_points).to(torch.float32) for img_points in gt_points_b] 

    gt_logits = [torch.zeros((img_points.shape[0], emb_size)) for img_points in gt_points] 

    
    tokenized = model.tokenizer(captions, padding="longest", return_tensors="pt")

    # find last index of special token (.)
    end_idxes = [torch.where(input_ids==1012)[0][-1] for input_ids in tokenized['input_ids']] 
    for i, end_idx in enumerate(end_idxes):
        gt_logits[i][:,:end_idx] = 1.0 

    caption_sizes = [end_idx + 2 for end_idx in end_idxes]  # incl. CLS and SEP

    targets = [{"points": img_gt_points.to(device), "labels": img_gt_logits.to(device), "caption_size": caption_size} for img_gt_points, img_gt_logits, caption_size in zip(gt_points, gt_logits, caption_sizes)] 

    return targets


def distance_threshold_func(boxes): # list of [xc,yc,w,h]
    if len(boxes) == 0:
        return 0.0
    # find median index of boxes areas
    areas = [box[2]*box[3] for box in boxes]
    median_idx = np.argsort(areas)[len(areas)//2]
    median_box = boxes[median_idx]
    w = median_box[2]
    h = median_box[3]

    threshold = np.sqrt(w**2 + h**2) / 2.0
    
    return threshold

def calc_loc_metric(pred_boxes, gt_points): # list of [xc,yc,w,h], tensor of (nt,2)
    if len(pred_boxes) == 0:
        FN = len(gt_points)
        return 0, 0, FN, 0, 0, 0
    
    dist_threshold = distance_threshold_func(pred_boxes)
    pred_points = np.array([[box[0], box[1]] for box in pred_boxes])
    gt_points = gt_points.cpu().detach().numpy()

    # create a cost matrix
    cost_matrix = cdist(pred_points, gt_points, metric='euclidean')
    
    # use Hungarian algorithm to find optimal assignment
    pred_indices, gt_indices = linear_sum_assignment(cost_matrix)
    
    # determine TP, FP, FN
    TP = 0
    for pred_idx, gt_idx in zip(pred_indices, gt_indices):
        if cost_matrix[pred_idx, gt_idx] < dist_threshold:
            TP += 1
    
    FP = len(pred_points) - TP
    FN = len(gt_points) - TP

    Precision = TP / (TP + FP) if TP + FP != 0 else 0.0
    Recall = TP / (TP + FN) if TP + FN != 0 else 0.0
    F1 = 2 * (Precision * Recall) / (Precision + Recall) if Precision + Recall != 0 else 0.0
    
    return TP, FP, FN, Precision, Recall, F1
