import logging
from typing import Tuple
from transformers import (
    AutoModelForSequenceClassification, AutoConfig,
    AutoTokenizer, PreTrainedTokenizer
)

from .abc import AbsRerankerRunner, AbsRerankerModel
from .modeling import CrossEncoderModel
from .trainer import EncoderOnlyRerankerTrainer

logger = logging.getLogger(__name__)


class EncoderOnlyRerankerRunner(AbsRerankerRunner):
    """
    Encoder only reranker runner for finetuning.
    """
    def load_tokenizer_and_model(self) -> Tuple[PreTrainedTokenizer, AbsRerankerModel]:
        """Load the tokenizer and model.

        Args:
            self: 实例对象
            
        Returns:
            Tuple[PreTrainedTokenizer, AbsRerankerModel]: Tokenizer and model instances.
        """
        # 加载预训练分词器
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_args.model_name_or_path,
            cache_dir=self.model_args.cache_dir,
            token=self.model_args.token,
            trust_remote_code=self.model_args.trust_remote_code
        )

        # 设置分类标签数量为1（用于二分类或回归任务）
        num_labels = 1
        # 加载模型配置
        config = AutoConfig.from_pretrained(
            self.model_args.config_name if self.model_args.config_name else self.model_args.model_name_or_path,
            num_labels=num_labels,
            cache_dir=self.model_args.cache_dir,
            token=self.model_args.token,
            trust_remote_code=self.model_args.trust_remote_code,
        )
        logger.info('Config: %s', config)

        # 加载序列分类基础模型
        base_model = AutoModelForSequenceClassification.from_pretrained(
            self.model_args.model_name_or_path,
            config=config,
            cache_dir=self.model_args.cache_dir,
            token=self.model_args.token,
            from_tf=bool(".ckpt" in self.model_args.model_name_or_path),
            trust_remote_code=self.model_args.trust_remote_code
        )

        # 构建交叉编码器模型
        model = CrossEncoderModel(
            base_model,
            tokenizer=tokenizer,
            train_batch_size=self.training_args.per_device_train_batch_size,
        )

        # 如果启用梯度检查点，则启用输入梯度计算
        if self.training_args.gradient_checkpointing:
            model.enable_input_require_grads()

        return tokenizer, model

    def load_trainer(self) -> EncoderOnlyRerankerTrainer:
        """Load the trainer.

        Args:
            self: 实例对象
            
        Returns:
            EncoderOnlyRerankerTrainer: Loaded trainer instance.
        """
        # 创建并返回重排序训练器实例
        trainer = EncoderOnlyRerankerTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer
        )
        return trainer