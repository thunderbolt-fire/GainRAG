# python
import torch
from torch import nn, Tensor
from transformers import AutoTokenizer
from transformers.file_utils import ModelOutput

import torch.nn.functional as F

# python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename="grpo.log",
    filemode="w",  # 每次覆盖，想追加就用 "a"
)

from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Dict, Optional, List, Union, Any

logger = logging.getLogger(__name__)


@dataclass
class RerankerOutput(ModelOutput):
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None

    # 下面是新增的一些调试 / 统计信息
    raw_teacher_scores: Optional[Tensor] = None      # (B, K)
    grpo_rewards: Optional[Tensor] = None            # (B, K)
    grpo_advantage: Optional[Tensor] = None          # (B, K)
    grpo_kl_loss: Optional[Tensor] = None            # 标量
    grpo_entropy: Optional[Tensor] = None            # 标量


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
        grpo_temperature: float = 1.0,   # teacher->reward 的温度
        grpo_normalize: bool = True,     # 是否用组内标准差归一化 advantage
        grpo_clip_adv: float = 5.0,      # 裁剪 advantage，0 或 None 关闭
        grpo_kl_coef: float = 0.0,       # 可选 KL 正则系数，0 关闭
        entropy_coef: float = 0.0,       # 可选熵奖励系数，>0 提升探索
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

        # GRPO 超参
        self.grpo_temperature = grpo_temperature
        self.grpo_normalize = grpo_normalize
        self.grpo_clip_adv = grpo_clip_adv
        self.grpo_kl_coef = grpo_kl_coef
        self.entropy_coef = entropy_coef

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

    def _grpo_loss(
        self,
        student_log_probs: Tensor,  # (B, K), 即 log_softmax(logits)
        teacher_scores: Tensor,     # (B, K)
    ) -> (Tensor, Dict[str, Tensor]):
        """
        返回：
          pg_loss: 标量 loss
          stats:   各种中间统计信息
        """
        # 保存一份原始 teacher_scores（detach 避免反传）
        raw_teacher_scores = teacher_scores.detach()

        # 1) teacher_scores -> 组内奖励分布（softmax + 温度）
        tau = max(self.grpo_temperature, 1e-6)
        rewards = torch.softmax(teacher_scores / tau, dim=-1)  # (B, K), ∈ (0,1), sum=1

        # 2) 组内相对优势（Group Relative Advantage）
        adv = rewards - rewards.mean(dim=-1, keepdim=True)
        if self.grpo_normalize:
            std = rewards.std(dim=-1, keepdim=True).clamp_min(1e-8)
            adv = adv / std

        if self.grpo_clip_adv and self.grpo_clip_adv > 0:
            adv = adv.clamp(min=-self.grpo_clip_adv, max=self.grpo_clip_adv)

        # 3) REINFORCE 风格的目标：- E[adv * log pi(a)]
        # 注意：adv 只作为权重，不反传
        pg_loss = - (adv.detach() * student_log_probs).sum(dim=-1).mean()

        # 4) 可选：KL 正则到 teacher 分布，稳定训练（默认 0 关闭）
        kl_loss = torch.tensor(0.0, device=student_log_probs.device)
        if self.grpo_kl_coef and self.grpo_kl_coef > 0:
            teacher_targets = rewards.detach()  # teacher 的 soft 分布
            kl_loss = F.kl_div(student_log_probs, teacher_targets, reduction="batchmean")
            pg_loss = pg_loss + self.grpo_kl_coef * kl_loss

        # 5) 可选：熵奖励，提升探索（默认 0 关闭）
        entropy = torch.tensor(0.0, device=student_log_probs.device)
        if self.entropy_coef and self.entropy_coef > 0:
            probs = student_log_probs.exp()
            entropy = -(probs * student_log_probs).sum(dim=-1).mean()
            pg_loss = pg_loss - self.entropy_coef * entropy

        stats = {
            "raw_teacher_scores": raw_teacher_scores,  # (B, K)
            "grpo_rewards": rewards.detach(),          # (B, K)
            "grpo_advantage": adv.detach(),            # (B, K)
            "grpo_kl_loss": kl_loss.detach(),          # 标量
            "grpo_entropy": entropy.detach(),          # 标量
        }

        # 可选：也可以 log 一下平均值 + 前几项，便于对比分数变化
        with torch.no_grad():
            # 展平到 1 维后取前 8 个元素做示例
            flat_teacher = raw_teacher_scores.view(-1)
            flat_rewards = rewards.view(-1)

            logger.debug(
                "GRPO stats: "
                f"teacher_scores_mean={raw_teacher_scores.mean().item():.4f}, "
                f"rewards_mean={rewards.mean().item():.4f}, "
                f"adv_mean={adv.mean().item():.4f}, "
                f"kl_loss={kl_loss.item():.4f}, "
                f"entropy={entropy.item():.4f}\n"
                f"  teacher_scores_sample={flat_teacher[:8].tolist()}\n"
                f"  grpo_rewards_sample={flat_rewards[:8].tolist()}"
            )

        return pg_loss, stats

    def forward(
        self,
        pair: Union[Dict[str, Tensor], List[Dict[str, Tensor]]] = None,
        teacher_scores: Optional[Tensor] = None,
    ):
        """The computation performed at every call.

        Args:
            pair (Union[Dict[str, Tensor], List[Dict[str, Tensor]]], optional): The query-document pair. Defaults to ``None``.
            teacher_scores (Optional[Tensor], optional): Teacher scores of knowledge distillation. Defaults to None.

        Returns:
            RerankerOutput: Output of reranker model.
        """
        ranker_logits = self.encode(pair)  # (B*K, 1 or dim) -> 将在下面 reshape

        raw_teacher_scores = None
        grpo_rewards = None
        grpo_advantage = None
        grpo_kl_loss = None
        grpo_entropy = None

        if self.training:
            grouped_logits = ranker_logits.view(self.train_batch_size, -1)  # (B, K)

            # 准备 teacher_scores 与 student_inputs
            teacher_scores = torch.as_tensor(
                teacher_scores, device=grouped_logits.device
            )
            teacher_scores = teacher_scores.view(self.train_batch_size, -1)  # (B, K)

            student_log_probs = torch.log_softmax(grouped_logits, dim=-1)   # (B, K)

            # 使用 GRPO 的组相对策略梯度损失，同时拿到统计信息
            loss, stats = self._grpo_loss(student_log_probs, teacher_scores)

            raw_teacher_scores = stats["raw_teacher_scores"]
            grpo_rewards = stats["grpo_rewards"]
            grpo_advantage = stats["grpo_advantage"]
            grpo_kl_loss = stats["grpo_kl_loss"]
            grpo_entropy = stats["grpo_entropy"]
        else:
            loss = None

        return RerankerOutput(
            loss=loss,
            scores=ranker_logits,
            raw_teacher_scores=raw_teacher_scores,
            grpo_rewards=grpo_rewards,
            grpo_advantage=grpo_advantage,
            grpo_kl_loss=grpo_kl_loss,
            grpo_entropy=grpo_entropy,
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