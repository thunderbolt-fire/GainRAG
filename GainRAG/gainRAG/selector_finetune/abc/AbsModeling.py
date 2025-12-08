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

    def forward(self, pair: Union[Dict[str, Tensor], List[Dict[str, Tensor]]] = None, teacher_scores: Optional[Tensor] = None):
        """
        GRPO-style training for Reranker using teacher scores as rewards.
        """
        # 1. 获取 Student 模型的原始打分 (Logits)
        ranker_logits = self.encode(pair) # (batch_size * num_docs_per_query, )
        
        if self.training:
            # 2. 重塑形状为 (Batch_Size, Group_Size)
            # Group_Size 就是每个 Query 对应的文档数量
            grouped_logits = ranker_logits.view(self.train_batch_size, -1)
            
            # 3. 处理 Teacher Scores (作为 Reward)
            # 确保 teacher_scores 是 Tensor 且在正确的设备上
            if not isinstance(teacher_scores, torch.Tensor):
                teacher_scores = torch.tensor(teacher_scores)
            rewards = teacher_scores.view(self.train_batch_size, -1).to(grouped_logits.device)

            # --- GRPO 核心逻辑 ---

            # 4. 计算组内优势 (Advantage)
            # 对每个 Query (Group) 内部计算 Reward 的均值和标准差
            mean_rewards = rewards.mean(dim=-1, keepdim=True)
            std_rewards = rewards.std(dim=-1, keepdim=True)
            
            # 标准化 Reward 得到 Advantage
            # 加上 1e-8 防止除以零
            advantages = (rewards - mean_rewards) / (std_rewards + 1e-8)
            
            # 5. 计算 Student 的策略分布 (Log Probabilities)
            # 使用 Log Softmax 将 logits 转化为对数概率
            student_log_probs = torch.log_softmax(grouped_logits, dim=-1)
            
            # 6. 计算 GRPO Loss
            # 目标：最大化 E[log_prob * advantage]
            # 损失：最小化 - (log_prob * advantage)
            # detach() 很重要：我们不希望梯度传导回 advantage (即不更新 teacher 或 reward 计算方式)
            
            # 逐元素相乘，然后在 Group 维度求和，最后在 Batch 维度求平均
            loss = -torch.mean(torch.sum(student_log_probs * advantages.detach(), dim=-1))
            
            # 可选：添加 KL 散度正则项 (防止 Student 策略偏离 Teacher 太远，如果需要的话)
            # ref_log_probs = torch.log_softmax(rewards, dim=-1) # 假设 teacher scores 也是 logits
            # kl_loss = torch.nn.functional.kl_div(student_log_probs, ref_log_probs, reduction='batchmean', log_target=True)
            # loss = loss + beta * kl_loss

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