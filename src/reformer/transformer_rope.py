from transformer.config import DataConfig, ModelConfig
from transformer.transformer import Transformer

from .components.multi_head_rope import MultiHeadAttentionRoPE


class TransformerRoPE(Transformer):
    """Transformer with RoPE - reuses all existing config parameters"""

    def __init__(self, model_config: ModelConfig, data_config: DataConfig):
        super().__init__(
            model_config=model_config,
            data_config=data_config,
            attention_cls=MultiHeadAttentionRoPE,
            positional_encoding_cls=None,
            use_input_positional_encoding=False,
            src_max_len=model_config.src_max_len,
            tgt_max_len=model_config.tgt_max_len,
            base=10000.0,
        )
