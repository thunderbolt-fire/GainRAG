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
        ranker_logits = self.encode(pair) # (batch_size * num, dim)  # dim=1, num=len(pairs 一个item组成的)


        if self.training:
            # ranker_logits: (B*N, 1) 或 (B*N,)
            if ranker_logits.dim() == 2 and ranker_logits.size(-1) == 1:
                ranker_logits = ranker_logits.squeeze(-1)  # -> (B*N,)

            # teacher_scores: list 或 tensor, 长度应为 B*N
            teacher_scores = torch.as_tensor(
                teacher_scores,
                dtype=ranker_logits.dtype,
                device=ranker_logits.device
            ).view(-1)

            total = ranker_logits.numel()
            if teacher_scores.numel() != total:
                raise ValueError(f"teacher_scores numel({teacher_scores.numel()}) != logits numel({total})")

            # N = train_group_size（优先使用显式配置；否则兜底用 train_batch_size 推断）
            N = getattr(self, "train_group_size", None)
            if N is None:
                tb = int(getattr(self, "train_batch_size", 0) or 0)
                if tb <= 0:
                    raise ValueError("Missing model.train_group_size and invalid model.train_batch_size for fallback.")
                # 兜底假设：当前 batch 的 query 数 == self.train_batch_size
                if total % tb != 0:
                    raise ValueError(f"(B*N)={total} not divisible by fallback train_batch_size(B)={tb}; "
                                    f"please set model.train_group_size = data_args.train_group_size")
                N = total // tb  # 推断 group_size

            # 校验 N 合法
            if N <= 0:
                raise ValueError(f"Invalid train_group_size(N)={N}")

            if total % N != 0:
                raise ValueError(f"(B*N)={total} not divisible by train_group_size(N)={N}")

            B = total // N

            grouped_logits = ranker_logits.view(B, N)      # (B, N)
            teacher_scores = teacher_scores.view(B, N)     # (B, N)

            # === 下面保持你原来的 loss 计算不变 ===
            teacher_probs = torch.softmax(teacher_scores.detach(), dim=-1)
            logp = torch.log_softmax(grouped_logits, dim=-1)
            p = torch.softmax(grouped_logits, dim=-1)

            kd_loss = F.kl_div(logp, teacher_probs, reduction="batchmean")

            adv = teacher_probs - teacher_probs.mean(dim=-1, keepdim=True)
            adv = adv / (adv.std(dim=-1, keepdim=True) + 1e-8)
            adv = adv.clamp(-5.0, 5.0).detach()

            grpo_loss = -(adv * logp).sum(dim=-1).mean()

            kl_pi_teacher = (p * (torch.log(p + 1e-12) - torch.log(teacher_probs + 1e-12))).sum(dim=-1).mean()

            kd_w = getattr(self, "kd_weight", 1.0)
            grpo_w = getattr(self, "grpo_weight", 1.0)
            kl_w = getattr(self, "kl_weight", 0.1)

            loss = kd_w * kd_loss + grpo_w * grpo_loss + kl_w * kl_pi_teacher
        else:
            loss = None

        return RerankerOutput(
            loss=loss,
            scores=ranker_logits,
        )

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