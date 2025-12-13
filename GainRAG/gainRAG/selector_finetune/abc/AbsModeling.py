import torch
from torch import nn, Tensor
from transformers import AutoTokenizer
from transformers.file_utils import ModelOutput

import torch.nn.functional as F

import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Dict, Optional, List, Union

logger = logging.getLogger(__name__)


@dataclass
class RerankerOutput(ModelOutput):
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None


class AbsRerankerModel(ABC, nn.Module):
    """Abstract class of embedding model for training.

    Args:
        base_model: The base model to train on.
        tokenizer (AutoTokenizer, optional): The tokenizer to use. Defaults to ``None``.
        train_batch_size (int, optional): Batch size used for training. Defaults to ``4``.
    """
    def __init__(
        self,
        base_model: None,
        tokenizer: AutoTokenizer = None,
        train_batch_size: int = 4,
    ):
        super().__init__()
        self.model = base_model
        self.tokenizer = tokenizer
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.config = self.model.config

        self.train_batch_size = train_batch_size

        self.yes_loc = self.tokenizer('Yes', add_special_tokens=False)['input_ids'][-1]

    def gradient_checkpointing_enable(self, **kwargs):
        """
        Activates gradient checkpointing for the current model.
        """
        self.model.gradient_checkpointing_enable(**kwargs)

    def enable_input_require_grads(self, **kwargs):
        """
        Enables the gradients for the input embeddings.
        """
        self.model.enable_input_require_grads(**kwargs)

    @abstractmethod
    def encode(self, features):
        """Abstract method of encode.

        Args:
            features (dict): Teatures to pass to the model.
        """
        pass

    def forward(self, pair=None, teacher_scores=None):
        ranker_logits = self.encode(pair)  # (B*N, 1) 或 (B*N,)
        if ranker_logits.dim() == 2 and ranker_logits.size(-1) == 1:
            ranker_logits = ranker_logits.squeeze(-1)  # (B*N,)

        if not self.training:
            return RerankerOutput(loss=None, scores=ranker_logits)

        # ---------- 0) 动态推断 B 和 N，避免 self.train_batch_size 不一致导致错位 ----------
        teacher_scores = torch.as_tensor(teacher_scores, dtype=ranker_logits.dtype, device=ranker_logits.device)
        total = ranker_logits.numel()
        if teacher_scores.numel() != total:
            raise ValueError(f"teacher_scores numel({teacher_scores.numel()}) != logits numel({total})")

        # 训练组大小 N：优先从 args 或模型属性拿（你数据集里是 args.train_group_size）
        group_size = getattr(self, "train_group_size", None)
        if group_size is None:
            # 兜底：尝试从固定 train_group_size 推断；否则只能用 self.train_batch_size
            # 建议你在构建模型时：self.train_group_size = data_args.train_group_size
            group_size = total // getattr(self, "train_batch_size", 1)

        if total % group_size != 0:
            raise ValueError(f"(B*N)={total} not divisible by group_size={group_size}")

        B = total // group_size
        N = group_size

        grouped_logits = ranker_logits.view(B, N)          # (B, N)
        teacher_scores = teacher_scores.view(B, N)         # (B, N)

        # ---------- 1) teacher 分布：分数在[0,1]时必须加温度，否则接近均匀 ----------
        t_temp = float(getattr(self, "teacher_temp", 0.2))  # 建议 0.1~0.5
        t_logits = teacher_scores.detach() / max(t_temp, 1e-6)
        teacher_probs = torch.softmax(t_logits, dim=-1)     # (B, N)

        # student 分布
        logp = torch.log_softmax(grouped_logits, dim=-1)    # (B, N)
        p = torch.softmax(grouped_logits, dim=-1)           # (B, N)

        # ---------- 2) KD：KL(teacher || student) >= 0 ----------
        kd_loss = F.kl_div(logp, teacher_probs, reduction="batchmean")

        # ---------- 3) GRPO（全动作加权版本：按组内 advantage 加权 logp） ----------
        tlogp = torch.log(teacher_probs + 1e-12)  # 保留 teacher_probs 原样
        adv = tlogp - tlogp.mean(dim=-1, keepdim=True)          # (B, N)
        adv = adv / (adv.std(dim=-1, keepdim=True) + 1e-8)      # (B, N)
        adv = adv.clamp(-5.0, 5.0).detach()                     # (B, N)

        grpo_loss = -(adv * logp).sum(dim=-1).mean()

        # 熵正则（可选，防止策略过早塌缩导致梯度尖峰）
        entropy = -(p * logp).sum(dim=-1).mean()
        ent_w = float(getattr(self, "entropy_weight", 0.0))  # 建议先 0.0~0.01
        grpo_loss = grpo_loss - ent_w * entropy

        # ---------- 4) KL 约束：KL(student || teacher) >= 0 ----------
        kl_pi_teacher = (p * (torch.log(p + 1e-12) - torch.log(teacher_probs + 1e-12))).sum(dim=-1).mean()

        # 权重（建议 grpo 小一点起步）
        kd_w = float(getattr(self, "kd_weight", 1.0))
        grpo_w = float(getattr(self, "grpo_weight", 0.1))
        kl_w = float(getattr(self, "kl_weight", 0.1))

        loss = kd_w * kd_loss + grpo_w * grpo_loss + kl_w * kl_pi_teacher

        # ---------- 5) 数值保护：出现 NaN/Inf 就退化到纯 KD ----------
        if not torch.isfinite(loss):
            logger.warning("Non-finite loss detected, fallback to kd_loss only.")
            loss = kd_loss

        return RerankerOutput(loss=loss, scores=ranker_logits)

    def compute_loss(self, scores, target):
        """Compute the loss.

        Args:
            scores (torch.Tensor): Computed scores.
            target (torch.Tensor): The target value.

        Returns:
            torch.Tensor: The computed loss.
        """
        return self.cross_entropy(scores, target)

    def save(self, output_dir: str):
        """Save the model.

        Args:
            output_dir (str): Directory for saving the model.
        """
        # self.model.save_pretrained(output_dir)
        state_dict = self.model.state_dict()
        state_dict = type(state_dict)(
            {k: v.clone().cpu()
             for k,
             v in state_dict.items()})
        self.model.save_pretrained(output_dir, state_dict=state_dict)

    def save_pretrained(self, *args, **kwargs):
        """
        Save the tokenizer and model.
        """
        self.tokenizer.save_pretrained(*args, **kwargs)
        return self.model.save_pretrained(*args, **kwargs)