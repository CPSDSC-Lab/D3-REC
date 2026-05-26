import os
import torch
import torch.nn.functional as F


def _normalize_feature(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    if x.ndim == 1:
        x = x.unsqueeze(0)
    return F.normalize(x, dim=-1)


def extract_query_features(outputs):
    img_feat = outputs.get("img_embs", None)
    txt_feat = outputs.get("txt_embs", None)
    if img_feat is None:
        img_feat = outputs["pred_logits"].detach().sigmoid().mean(dim=1)
    elif img_feat.ndim == 3:
        img_feat = img_feat.detach().mean(dim=1)
    else:
        img_feat = img_feat.detach()
    if txt_feat is None:
        txt_feat = outputs["pred_logits"].detach().sigmoid().mean(dim=1)
    elif txt_feat.ndim == 3:
        txt_feat = txt_feat.detach().mean(dim=1)
    else:
        txt_feat = txt_feat.detach()
    return _normalize_feature(img_feat), _normalize_feature(txt_feat)


class SupportMemory:
    def __init__(self, img_feat, txt_feat, t1, t2, k_bias, device="cpu"):
        self.device = torch.device(device)
        self.img_feat = _normalize_feature(img_feat).to(self.device)
        self.txt_feat = _normalize_feature(txt_feat).to(self.device)
        self.t1 = t1.float().to(self.device)
        self.t2 = t2.float().to(self.device)
        self.k_bias = k_bias.float().to(self.device)

    @property
    def size(self):
        return int(self.img_feat.shape[0])

    @classmethod
    def load(cls, path, device="cpu"):
        if path is None or not os.path.exists(path):
            return None
        data = torch.load(path, map_location="cpu")
        img_feat = data["img_feat"]
        txt_feat = data["txt_feat"]
        t1 = data.get("t1", torch.full((img_feat.shape[0],), 0.25))
        t2 = data.get("t2", torch.full((img_feat.shape[0],), 0.35))
        k_bias = data.get("k_bias", torch.zeros((img_feat.shape[0],)))
        return cls(img_feat, txt_feat, t1, t2, k_bias, device=device)

    def retrieve(self, q_img_feat, q_txt_feat, topk=8, alpha=0.5, tau=0.07):
        if self.size == 0:
            return None
        q_img_feat = _normalize_feature(q_img_feat).to(self.device)
        q_txt_feat = _normalize_feature(q_txt_feat).to(self.device)
        sim_img = torch.matmul(q_img_feat, self.img_feat.T)
        sim_txt = torch.matmul(q_txt_feat, self.txt_feat.T)
        sim = alpha * sim_img + (1 - alpha) * sim_txt
        k = min(int(topk), self.size)
        if k <= 0:
            return None
        values, indices = torch.topk(sim, k, dim=1)
        weights = torch.softmax(values / max(float(tau), 1e-4), dim=1)
        t1_prior = torch.sum(self.t1[indices] * weights, dim=1)
        t2_prior = torch.sum(self.t2[indices] * weights, dim=1)
        k_bias_prior = torch.sum(self.k_bias[indices] * weights, dim=1)
        return {
            "t1_prior": t1_prior,
            "t2_prior": t2_prior,
            "k_bias_prior": k_bias_prior,
        }


def save_support_memory(path, records):
    if len(records) == 0:
        return
    img_feat = torch.stack([r["img_feat"] for r in records], dim=0).cpu()
    txt_feat = torch.stack([r["txt_feat"] for r in records], dim=0).cpu()
    t1 = torch.tensor([r["t1"] for r in records], dtype=torch.float32)
    t2 = torch.tensor([r["t2"] for r in records], dtype=torch.float32)
    k_bias = torch.tensor([r["k_bias"] for r in records], dtype=torch.float32)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "img_feat": img_feat,
            "txt_feat": txt_feat,
            "t1": t1,
            "t2": t2,
            "k_bias": k_bias,
        },
        path,
    )
