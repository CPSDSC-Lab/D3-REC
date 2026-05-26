import os
import sys
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
GDDINO_DIR = os.path.join(ROOT_DIR, "GroundingDINO")
if GDDINO_DIR not in sys.path:
    sys.path.insert(0, GDDINO_DIR)
import torch
import numpy as np
import copy
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from groundingdino.util.base_api import infer_config_overrides_from_checkpoint, load_model, threshold
import os
import numpy as np
from datetime import datetime

from utils.processor import DataProcessor
from utils.criterion import SetCriterion, L2Loss, SetRegContrastiveCriterion
from utils.image_loader import get_loader
from tqdm import tqdm
from utils.util import visualize_and_save_points, visualize_density_map

from utils.trainer_rec import *
import random
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import argparse

parser = argparse.ArgumentParser()
## training setting
parser.add_argument("--epochs", default=1, type=int, help="epoches")
parser.add_argument("--un_epochs", default=15, type=int, help="epoches")
parser.add_argument("--batch", default=1, type=int, help="batch size")
parser.add_argument("--seed", default=314, type=int, help="batch size")
parser.add_argument("--scale", default=1000, type=int, help="batch size")

## model setting 
parser.add_argument('--config', default="./GroundingDINO/groundingdino/config/GroundingDINO_SwinT_density_guide.py", type=str, help='pretrain pth')
parser.add_argument('--pretrain_model', default="./exp/cvpr/sdpr_only_fixed/model.pth", type=str, help='pretrain pth')
parser.add_argument('--load_mode', default="test", type=str, help='loading mode')
parser.add_argument('--prompt_detach', action='store_true', help="if detach the prompt and density feature")
parser.add_argument('--use_ddg', action='store_true', default=False, help='override to enable DDG')
parser.add_argument('--use_adm', action='store_true', default=False, help='override to enable ADM')
parser.add_argument('--use_sdpr', action='store_true', default=False, help='override to enable SDPR')
parser.add_argument('--num_attr_slots', type=int, default=4, help='number of ADM slots')
parser.add_argument('--sd_attn_dir', type=str, default=None, help='directory of precomputed SD attention maps')

## saving setting
parser.add_argument("--stats_dir", default="./exp/cvpr/sdpr_only_fixed/test_results", type=str, help='stats directory')
parser.add_argument("--vis_dir", default="localization_test", type=str, help='stats directory')
parser.add_argument("--vis_density_dir", default="density", type=str, help='stats directory')
parser.add_argument("--result_txt", default="result.txt", type=str, help='stats directory')
parser.add_argument("--selection_txt", default="result_err_20_test.txt", type=str, help='stats directory')
parser.add_argument("--write_txt", action='store_true', help="write txt result")
parser.add_argument("--write_vis", action='store_true', help="write visual result")
parser.add_argument("--write_density", action='store_true', help="write density or not")
## test setting 
parser.add_argument("--pred_num_judge", default=650, type=int, help='patch threshold')
parser.add_argument("--threshold1", default=0.25, type=float, help='threshold for localization')
parser.add_argument("--threshold2", default=0.35, type=float, help='threshold for localization')
parser.add_argument("--eval_split", default="test", type=str, choices=["train", "val", "test"], help='evaluation split')
parser.add_argument("--eval_impl", default="eval_fn_dagger", type=str, choices=["eval_fn", "eval_fn_dagger", "eval_fn_dagger_calibrated", "eval_fn_hybrid_local"], help='evaluation implementation')
parser.add_argument("--dagger_calib_a", default=1.0, type=float, help='calibration scale for calibrated dagger top-k')
parser.add_argument("--dagger_calib_b", default=0.0, type=float, help='calibration bias for calibrated dagger top-k')
parser.add_argument("--hybrid_grid_rows", default=4, type=int, help='local hybrid grid rows')
parser.add_argument("--hybrid_grid_cols", default=4, type=int, help='local hybrid grid cols')
parser.add_argument("--hybrid_sparse_th", default=1.5, type=float, help='use detection count when local density count <= this threshold')
parser.add_argument("--hybrid_dense_th", default=4.0, type=float, help='use regression count when local density count >= this threshold')
parser.add_argument("--use_support", action='store_true', help='enable support-adaptive inference')
parser.add_argument("--support_path", default="", type=str, help='support memory path')
parser.add_argument("--support_collect_path", default="", type=str, help='path to save collected support memory')
parser.add_argument("--support_topk", default=8, type=int, help='retrieval topk')
parser.add_argument("--support_alpha", default=0.5, type=float, help='image-text retrieval blend')
parser.add_argument("--support_tau", default=0.07, type=float, help='retrieval softmax temperature')
parser.add_argument("--support_gamma", default=0.6, type=float, help='prior blend weight')
parser.add_argument("--support_k_min", default=1, type=int, help='min dynamic topk')
parser.add_argument("--support_k_max", default=900, type=int, help='max dynamic topk')
parser.add_argument("--support_k_bias_clip", default=20.0, type=float, help='k bias clip')
parser.add_argument("--support_t1_min", default=0.1, type=float, help='min threshold1')
parser.add_argument("--support_t1_max", default=0.6, type=float, help='max threshold1')
parser.add_argument("--support_t2_min", default=0.1, type=float, help='min threshold2')
parser.add_argument("--support_t2_max", default=0.7, type=float, help='max threshold2')
parser.add_argument("--support_conf_scale", default=0.2, type=float, help='confidence-driven support scale')
parser.add_argument("--color_prefilter", action='store_true', help='enable HSV color prefilter in inference')
parser.add_argument("--color_prefilter_min_ratio", default=0.08, type=float, help='minimum color ratio inside candidate box')

args = parser.parse_args()

print(args)

""" seed fix """
seed_value = args.seed
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)

""" data """
processor = DataProcessor()
annotations = processor.annotations

BATCH_SIZE = args.batch
train_loader = get_loader(processor, 'train', BATCH_SIZE, sd_attn_dir=args.sd_attn_dir)
val_loader = get_loader(processor, 'val', BATCH_SIZE, sd_attn_dir=args.sd_attn_dir)
test_loader = get_loader(processor, 'test', BATCH_SIZE, sd_attn_dir=args.sd_attn_dir)

loaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}
print("Data loaded!")
print(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")


""" model"""
CONFIG_PATH = args.config
CHECKPOINT_PATH =args.pretrain_model
print(f"Inference on test set using best model: {CHECKPOINT_PATH}")
inferred_overrides = infer_config_overrides_from_checkpoint(CHECKPOINT_PATH)
resolved_use_ddg = inferred_overrides.get("use_ddg", False) or args.use_ddg or args.use_sdpr
resolved_use_sdpr = inferred_overrides.get("use_sdpr", False) or args.use_sdpr
config_overrides = {
    "use_ddg": resolved_use_ddg,
    "use_adm": args.use_adm,
    "use_sdpr": resolved_use_sdpr,
    "num_attr_slots": args.num_attr_slots,
}
print(
    f"Resolved switches -> inferred(ddg={inferred_overrides.get('use_ddg', False)}, "
    f"sdpr={inferred_overrides.get('use_sdpr', False)}), "
    f"final(ddg={resolved_use_ddg}, adm={args.use_adm}, sdpr={resolved_use_sdpr})"
)
if args.use_sdpr and not inferred_overrides.get("use_sdpr", False):
    print("Warning: checkpoint does not contain SDPR parameters; forcing use_sdpr may cause mismatch.")
model = load_model(CONFIG_PATH, CHECKPOINT_PATH, mode=args.load_mode, config_overrides=config_overrides)
model = model.to(device)
model.transformer.query_detach = args.prompt_detach

'''
这个是设置实验保存的位置，设置的是文件夹
'''
stats_dir = args.stats_dir
os.makedirs(stats_dir, exist_ok=True)

stats_file = f"{stats_dir}/stats.txt"
stats = list()

print(f"Saving stats to {stats_file}")

need_write_header = (not os.path.exists(stats_file)) or os.path.getsize(stats_file) == 0
if need_write_header:
    with open(stats_file, 'a') as f:
        header = ['train_mae', 'train_rmse', 'train_TP', 'train_FP', 'train_FN', 'train_precision', 'train_recall', 'train_f1', 'train_regression_mae','train_regression_rmse', 
                  '||', 'val_mae', 'val_rmse', 'val_TP', 'val_FP', 'val_FN', 'val_precision', 'val_recall', 'val_f1', 'val_regression_mae','val_regression_rmse',
                  '||', 'test_mae', 'test_rmse', 'test_TP', 'test_FP', 'test_FN', 'test_precision', 'test_recall', 'test_f1', 'test_regression_mae','test_regression_rmse']
        f.write("%s\n" % ' | '.join(header))
if args.eval_impl == "eval_fn":
    metrics = eval_fn(args.eval_split, model, loaders, annotations, args)
elif args.eval_impl == "eval_fn_dagger_calibrated":
    metrics = eval_fn_dagger_calibrated(args.eval_split, model, loaders, annotations, args)
elif args.eval_impl == "eval_fn_hybrid_local":
    metrics = eval_fn_hybrid_local(args.eval_split, model, loaders, annotations, args)
else:
    metrics = eval_fn_dagger(args.eval_split, model, loaders, annotations, args)
test_mae, test_rmse, test_TP, test_FP, test_FN, test_precision, test_recall, test_f1, test_mae_regression, test_rmse_regression = metrics
print(f"test MAE: {test_mae:5.2f}, RMSE: {test_rmse:5.2f}, TP: {test_TP}, FP: {test_FP}, FN: {test_FN}, precision: {test_precision:5.2f}, recall: {test_recall:5.2f}, f1: {test_f1:5.2f}, mae_regression: {test_mae_regression:5.2f}, mae_regression: {test_rmse_regression:5.2f}")
# write to stats file
line_inference = [0,0,0,0,0, 0,0,0, "||", 0,0,0,0,0, 0,0,0, "||", test_mae, test_rmse, test_TP, test_FP, test_FN, test_precision, test_recall, test_f1, test_mae_regression, test_rmse_regression]
with open(stats_file, 'a') as f:
    s = line_inference
    for i, x in enumerate(s):
        if type(x) != str:
            s[i] = str(round(x,4))
    f.write("%s\n" % ' | '.join(s))
