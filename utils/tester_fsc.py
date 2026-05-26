import os
import torch
import numpy as np
import copy
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from groundingdino.util.base_api import load_model, threshold, threshold_box
import os
import numpy as np
from datetime import datetime

from utils.processor import DataProcessor
from utils.criterion import SetCriterion, L2Loss, SetRegContrastiveCriterion
from utils.criterion_box import SetCriterionBox, SetCriterionFSC147
from utils.image_loader import get_loader
from utils.image_loader_fsc147 import get_fsc_loader
from tqdm import tqdm
import torchvision.transforms.functional as TF
import torchvision.transforms as T

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def _build_overlap_grid(h, w, rows, cols, overlap_ratio):
    rows = max(1, int(rows))
    cols = max(1, int(cols))
    overlap_ratio = max(0.0, float(overlap_ratio))
    y_edges = [round(i * h / rows) for i in range(rows + 1)]
    x_edges = [round(i * w / cols) for i in range(cols + 1)]
    crops = []
    for i in range(rows):
        for j in range(cols):
            base_top, base_bottom = y_edges[i], y_edges[i + 1]
            base_left, base_right = x_edges[j], x_edges[j + 1]
            pad_h = round((base_bottom - base_top) * overlap_ratio)
            pad_w = round((base_right - base_left) * overlap_ratio)
            top = max(0, base_top - pad_h)
            bottom = min(h, base_bottom + pad_h)
            left = max(0, base_left - pad_w)
            right = min(w, base_right + pad_w)
            crops.append((top, bottom, left, right))
    return crops


def _dedup_boxes_by_center(boxes, logits, h, w, radius_px):
    if boxes.numel() == 0 or boxes.shape[0] <= 1 or radius_px <= 0:
        return boxes, logits

    order = torch.argsort(logits, descending=True)
    boxes = boxes[order]
    logits = logits[order]

    centers = boxes[:, :2].clone()
    centers[:, 0] = centers[:, 0] * w
    centers[:, 1] = centers[:, 1] * h

    keep = []
    kept_centers = []
    radius_sq = float(radius_px) * float(radius_px)

    for idx in range(centers.shape[0]):
        center = centers[idx]
        is_duplicate = False
        for kept_center in kept_centers:
            dist_sq = torch.sum((center - kept_center) ** 2).item()
            if dist_sq <= radius_sq:
                is_duplicate = True
                break
        if not is_duplicate:
            keep.append(idx)
            kept_centers.append(center)

    keep = torch.tensor(keep, device=boxes.device, dtype=torch.long)
    return boxes[keep], logits[keep]

def threshold(
        outputs,
        captions: str,
        tokenizer,
        threshold1 = 0.25,
        threshold2 = 0.20): 

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
        mask2 = prediction_logits[:, 1:end_idx].gt(threshold2).all(dim=1) 
        mask = mask1 & mask2

        logits = prediction_logits[mask]  
        boxes = prediction_boxes[mask]  


        ret.append((boxes, logits.max(dim=1)[0]))

    return ret

def eval(split, model, loaders, args):
    print(f"Evaluation on {split} set")
    model.eval()
    loader = loaders[split]

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
    for images, patches, targets, boxes, texts, file_name in tqdm(loader): # tensor, list of list [caption] for each image in the batch, list, list of list [(img, cap)] for each img in the batch
        density_maps = targets['density_map'].squeeze(1).to(device)
        boxes = boxes.to(torch.float32).to(device)

        # points = targets['points'].to(device)
        images = images.to(device)
        with torch.no_grad():
            outputs = model(images, captions=texts)

        pred_density_map = outputs['density'].squeeze(1)

        pred_num = torch.sum(pred_density_map, dim=[1,2]) / args.scale
        gt_num_density = torch.sum(density_maps, dim = [1,2])
        
        for i, p in enumerate(gt_num_density):
            # gt_num = len(p['points'])
            cnt_err = torch.abs(pred_num[i] - gt_num_density[i])
            val_mae += cnt_err
            val_rmse += cnt_err ** 2
        
        density_maps.shape
        
        total_num += density_maps.shape[0]

        # localization target and loss
        outputs["pred_points"] = outputs["pred_boxes"][:, :, :2] 
        
        h, w = images.shape[-2:]
        shape = [h, w]
        emb_size = outputs["pred_logits"].shape[2]
        targets = prepare_targets(targets, emb_size, shape)

        counter_for_image += 1
        results = threshold_box(outputs, args.threshold)
        
        pred_num_judge = len(results[0][1])
        if pred_num_judge >= args.pred_num_judge:
            h, w = images.shape[-2:]
            r_images = []
            resize_transform = T.Resize(800)
            r_images.append(resize_transform(TF.crop(images[0], 0, 0, int(h/2), int(w/2))))
            r_images.append(resize_transform(TF.crop(images[0], int(h/2), 0, int(h/2), int(w/2))))
            r_images.append(resize_transform(TF.crop(images[0], 0, int(w/2), int(h/2), int(w/2))))
            r_images.append(resize_transform(TF.crop(images[0], int(h/2), int(w/2), int(h/2), int(w/2))))
            
            points_list = []
            logits_list = []
            density_pred_list = []
            for r_image in r_images:
                with torch.no_grad():
                    outputs = model(r_image.unsqueeze(0), captions=texts)
                    outputs["pred_points"] = outputs["pred_boxes"][:, :, :2] 
                    result_temp = threshold_box(outputs, threshold=args.threshold)
                    points_list.append(result_temp[0][0])
                    logits_list.append(result_temp[0][1])
                    density_pred_list.append(torch.sum(outputs['density'].squeeze(1), dim=[1,2]) / args.scale)
            points_1 = torch.cat(points_list, dim=0)
            logits_1 = torch.cat(logits_list, dim=0)
            new_pred_num_judge = len(logits_1)
            new_results = [[points_1, logits_1]]
            
            new_pred_density = sum(density_pred_list)

            if new_pred_density < 4 * pred_num:
                results = new_results
        
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
    
    eval_mae_regression = round(val_mae.item()/total_num,2)
    eval_rmse_regression = round(val_rmse.item()/total_num,2)**0.5

    eval_mae = eval_mae / total_num
    eval_rmse = (eval_rmse / total_num) ** 0.5

    eval_precision = eval_tp / (eval_tp + eval_fp) if eval_tp + eval_fp != 0 else 0.0
    eval_recall = eval_tp / (eval_tp + eval_fn) if eval_tp + eval_fn != 0 else 0.0
    eval_f1 = 2 * eval_precision * eval_recall / (eval_precision + eval_recall) if eval_precision + eval_recall != 0 else 0.0

    return eval_mae, eval_rmse, eval_tp, eval_fp, eval_fn, eval_precision, eval_recall, eval_f1, eval_mae_regression, eval_rmse_regression

def eval_reproduce(split, model, loaders, args):
    print(f"Evaluation on {split} set")
    model.eval()
    loader = loaders[split]

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
    for images, patches, targets, boxes, texts, file_name in tqdm(loader): # tensor, list of list [caption] for each image in the batch, list, list of list [(img, cap)] for each img in the batch
        # if total_num > 10:
        #     break
        density_maps = targets['density_map'].squeeze(1).to(device)
        boxes = boxes.to(torch.float32).to(device)

        # points = targets['points'].to(device)
        images = images.to(device)

        with torch.no_grad():
            outputs = model(images, captions=texts)
        pred_density_map = outputs['density'].squeeze(1)
        pred_num = torch.sum(pred_density_map, dim=[1,2]) / args.scale
        gt_num_density = torch.sum(density_maps, dim = [1,2])

        for i, _ in enumerate(gt_num_density):
            cnt_err = torch.abs(pred_num[i] - gt_num_density[i])
            val_mae += cnt_err
            val_rmse += cnt_err ** 2
        
        total_num += density_maps.shape[0]

        # localization target and loss
        outputs["pred_points"] = outputs["pred_boxes"][:, :, :2] 
        
        h, w = images.shape[-2:]
        shape = [h, w]
        emb_size = outputs["pred_logits"].shape[2]
        targets = prepare_targets(targets, emb_size, shape)

        counter_for_image += 1
        results = threshold_box(outputs, args.threshold)

        pred_num_judge = len(results[0][1])
        use_dense_path = pred_num_judge > args.pred_num_judge
        active_threshold = args.dense_threshold if use_dense_path else args.threshold
        if active_threshold != args.threshold:
            results = threshold_box(outputs, active_threshold)
            pred_num_judge = len(results[0][1])

        if use_dense_path:
            h, w = images.shape[-2:]
            resize_transform = T.Resize(800)
            crops = _build_overlap_grid(h, w, args.split_rows, args.split_cols, args.split_overlap)
            points_list = []
            logits_list = []
            for top, bottom, left, right in crops:
                crop_h = bottom - top
                crop_w = right - left
                r_image = resize_transform(TF.crop(images[0], top, left, crop_h, crop_w))
                with torch.no_grad():
                    outputs_crop = model(r_image.unsqueeze(0), captions=texts)
                    outputs_crop["pred_points"] = outputs_crop["pred_boxes"][:, :, :2]
                    result_temp = threshold_box(outputs_crop, threshold=active_threshold)

                crop_boxes, crop_logits = result_temp[0]
                if crop_boxes.numel() > 0:
                    crop_boxes = crop_boxes.clone()
                    crop_boxes[:, 0] = (crop_boxes[:, 0] * crop_w + left) / w
                    crop_boxes[:, 1] = (crop_boxes[:, 1] * crop_h + top) / h
                    crop_boxes[:, 2] = crop_boxes[:, 2] * crop_w / w
                    crop_boxes[:, 3] = crop_boxes[:, 3] * crop_h / h
                points_list.append(crop_boxes)
                logits_list.append(crop_logits)

            points_1 = torch.cat(points_list, dim=0)
            logits_1 = torch.cat(logits_list, dim=0)
            points_1, logits_1 = _dedup_boxes_by_center(
                points_1,
                logits_1,
                h,
                w,
                args.dedup_radius_px,
            )
            results = [[points_1, logits_1]]

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

    eval_mae_regression = round(val_mae.item() / total_num, 2)
    eval_rmse_regression = round(val_rmse.item() / total_num, 2) ** 0.5
    eval_mae = eval_mae / total_num
    eval_rmse = (eval_rmse / total_num) ** 0.5

    eval_precision = eval_tp / (eval_tp + eval_fp) if eval_tp + eval_fp != 0 else 0.0
    eval_recall = eval_tp / (eval_tp + eval_fn) if eval_tp + eval_fn != 0 else 0.0
    eval_f1 = 2 * eval_precision * eval_recall / (eval_precision + eval_recall) if eval_precision + eval_recall != 0 else 0.0

    return eval_mae, eval_rmse, eval_tp, eval_fp, eval_fn, eval_precision, eval_recall, eval_f1, eval_mae_regression, eval_rmse_regression


def _select_topk_results(results, topk_values):
    new_results = []
    for b, result in enumerate(results):
        boxes, logits = result
        logits = logits.reshape(-1)
        topk = int(topk_values[b])
        topk = max(1, topk)
        topk = min(topk, int(logits.numel()))
        if topk <= 0 or logits.numel() == 0:
            new_results.append((boxes[:0], logits[:0]))
            continue
        _, indices = torch.topk(logits, topk, dim=0)
        indices = indices.reshape(-1)
        selected_boxes = boxes.index_select(0, indices)
        selected_logits = logits.index_select(0, indices)
        new_results.append((selected_boxes, selected_logits))
    return new_results


def eval_dagger(split, model, loaders, args, calib_a=1.0, calib_b=0.0):
    print(f"Evaluation on {split} set")
    if calib_a != 1.0 or calib_b != 0.0:
        print(f"Using calibrated dagger: a={calib_a}, b={calib_b}")
    model.eval()
    loader = loaders[split]

    eval_mae = 0
    eval_rmse = 0

    val_mae = 0
    val_rmse = 0

    eval_tp = 0
    eval_fp = 0
    eval_fn = 0

    total_num = 0
    for images, patches, targets, boxes, texts, file_name in tqdm(loader):
        density_maps = targets['density_map'].squeeze(1).to(device)
        boxes = boxes.to(torch.float32).to(device)
        images = images.to(device)
        with torch.no_grad():
            outputs = model(images, captions=texts)

        pred_density_map = outputs['density'].squeeze(1)
        pred_num = torch.sum(pred_density_map, dim=[1, 2]) / args.scale
        gt_num_density = torch.sum(density_maps, dim=[1, 2])

        for i, _ in enumerate(gt_num_density):
            cnt_err = torch.abs(pred_num[i] - gt_num_density[i])
            val_mae += cnt_err
            val_rmse += cnt_err ** 2

        total_num += density_maps.shape[0]

        outputs["pred_points"] = outputs["pred_boxes"][:, :, :2]
        h, w = images.shape[-2:]
        shape = [h, w]
        emb_size = outputs["pred_logits"].shape[2]
        targets = prepare_targets(targets, emb_size, shape)

        results = threshold_box(outputs, args.threshold)
        topk_values = [round(calib_a * float(x.item()) + calib_b) for x in pred_num]
        results = _select_topk_results(results, topk_values)

        for b in range(len(results)):
            pred_boxes, pred_logits = results[b]
            pred_boxes = [box.tolist() for box in pred_boxes]

            points = [[box[0], box[1]] for box in pred_boxes]
            pred_cnt = len(points)
            gt_cnt = len(targets[b]["points"])
            cnt_err = abs(pred_cnt - gt_cnt)
            eval_mae += cnt_err
            eval_rmse += cnt_err ** 2

            TP, FP, FN, precision, recall, f1 = calc_loc_metric(pred_boxes, targets[b]["points"])
            eval_tp += TP
            eval_fp += FP
            eval_fn += FN

    eval_mae_regression = round(val_mae.item() / total_num, 2)
    eval_rmse_regression = round(val_rmse.item() / total_num, 2) ** 0.5
    eval_mae = eval_mae / total_num
    eval_rmse = (eval_rmse / total_num) ** 0.5

    eval_precision = eval_tp / (eval_tp + eval_fp) if eval_tp + eval_fp != 0 else 0.0
    eval_recall = eval_tp / (eval_tp + eval_fn) if eval_tp + eval_fn != 0 else 0.0
    eval_f1 = 2 * eval_precision * eval_recall / (eval_precision + eval_recall) if eval_precision + eval_recall != 0 else 0.0

    return eval_mae, eval_rmse, eval_tp, eval_fp, eval_fn, eval_precision, eval_recall, eval_f1, eval_mae_regression, eval_rmse_regression

def prepare_targets(anno_b, emb_size, shape):
    ## TODO: 生成对应的结果
    gt_points_b = [points/np.array(shape)[::-1] for points in anno_b['points']]
    gt_points = [torch.from_numpy(img_points).to(torch.float32) for img_points in gt_points_b] 
    gt_logits = [torch.zeros((img_points.shape[0], emb_size)) for img_points in gt_points] 
    
    for i, l in enumerate(gt_logits):
        gt_logits[i][:,:3] = 1.0 
    
    caption_sizes = [torch.tensor(3)] * len(gt_logits)
    
    targets = [{"points": img_gt_points.to(device), "labels": img_gt_logits.to(device), "caption_size":caption_size} for img_gt_points, img_gt_logits, caption_size in zip(gt_points, gt_logits, caption_sizes)] 

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
