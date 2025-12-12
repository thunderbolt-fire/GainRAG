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
        ranker_logits = self.encode(pair) 
        
        loss = None
        
        if self.training:
            # 2. 重塑形状
            grouped_logits = ranker_logits.view(self.train_batch_size, -1)
            
            # 3. 处理 Teacher Scores
            if teacher_scores is not None:
                if not isinstance(teacher_scores, torch.Tensor):
                    teacher_scores = torch.tensor(teacher_scores)
                rewards = teacher_scores.view(self.train_batch_size, -1).to(grouped_logits.device)

                # --- GRPO 核心逻辑 ---

                # 4. 计算组内优势 (Advantage)
                mean_rewards = rewards.mean(dim=-1, keepdim=True)
                std_rewards = rewards.std(dim=-1, keepdim=True)
                
                # [关键修改 1] 增加 epsilon 防止除零，并对结果进行截断 (Clamp)
                # 针对你的数据，std 可能非常小，epsilon 设大一点 (1e-5)
                advantages = (rewards - mean_rewards) / (std_rewards + 1e-5)
                
                # 截断 Advantage，防止因 Teacher 分数过于接近导致的梯度爆炸
                advantages = torch.clamp(advantages, min=-3.0, max=3.0)
                
                # 5. 计算 Student 的策略分布
                student_log_probs = torch.log_softmax(grouped_logits, dim=-1)
                
                # 6. 计算 GRPO Loss
                grpo_loss = -torch.mean(torch.sum(student_log_probs * advantages.detach(), dim=-1))
                
                # [关键修改 2] 必须启用 KL 散度正则项
                # 因为你的 Teacher 分数已经是 0-1 的概率值，我们希望 Student 的分布逼近这个分布
                # 但直接对 0-1 分数做 log_softmax 物理意义不对（它们不是 logits）
                # 更好的做法是：将 Teacher Scores 视为目标概率分布（归一化后）
                
                # 将 Teacher Scores 归一化为概率分布 (Sum=1)
                target_probs = rewards / (rewards.sum(dim=-1, keepdim=True) + 1e-8)
                # 避免 log(0)
                target_log_probs = torch.log(target_probs + 1e-10)
                
                # 计算 KL 散度: KL(Target || Student) 或者 KL(Student || Target)
                # 这里使用 PyTorch 标准 KLDivLoss: input=log_probs, target=probs (or log_probs if log_target=True)
                # 我们希望 Student (input) 逼近 Target
                
                # beta 系数：控制 KL 的权重。建议 0.01 - 0.1
                beta = 0.1 
                
                kl_loss = torch.nn.functional.kl_div(student_log_probs, target_log_probs, reduction='batchmean', log_target=True)
                
                loss = grpo_loss + beta * kl_loss
            else:
                loss = torch.tensor(0.0, device=ranker_logits.device, requires_grad=True)

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