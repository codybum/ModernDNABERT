"""
ALiBi Attention implementation for BERT models.

This module provides a modified BERT self-attention mechanism that incorporates
the Attention with Linear Biases (ALiBi) approach, which helps models extrapolate
to longer sequence lengths than seen during training.
"""

import math
import torch
import logging
from typing import Optional, Tuple, Union

from transformers.models.bert.modeling_bert import BertSelfAttention

logger = logging.getLogger(__name__)


class BertSelfAttentionWithALiBi(BertSelfAttention):
    """
    BERT self-attention with ALiBi (Attention with Linear Biases).
    This implementation dynamically handles varying sequence lengths.

    ALiBi applies a linear bias to attention scores based on the distance between tokens,
    which helps the model generalize to longer sequences than seen during training.
    """

    def __init__(self, config, position_embedding_type=None):
        super().__init__(config, position_embedding_type)

        # Disable position embeddings since ALiBi replaces them
        self.position_embedding_type = None

        # Get maximum supported length from config or use a reasonable default
        self.max_supported_length = getattr(config, "max_supported_length", 4096)

        # Pre-compute slopes for each attention head
        # The slopes follow a geometric sequence based on the number of heads
        num_heads = self.num_attention_heads
        m = torch.arange(1, num_heads + 1, dtype=torch.float32)
        m = m * 8 / num_heads  # Scale factor based on the ALiBi paper
        self.register_buffer("slopes", 1.0 / (2 ** m))

        # We'll compute distances dynamically for efficiency with long sequences

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        """
        Dynamic ALiBi implementation that works with any sequence length.

        Args:
            hidden_states: Input tensor of shape [batch_size, seq_length, hidden_size]
            attention_mask: Attention mask tensor
            head_mask: Mask for attention heads
            encoder_hidden_states: Optional encoder states for cross-attention
            encoder_attention_mask: Optional encoder attention mask for cross-attention
            past_key_value: Optional cached key/value tensor for incremental decoding
            output_attentions: Whether to return attention weights

        Returns:
            tuple: Output tensors, optionally including attention weights
        """
        # Get dimensions
        batch_size, seq_length, hidden_dim = hidden_states.shape

        # Log if sequence is very long
        if seq_length > self.max_supported_length:
            logger.warning(
                f"Sequence length ({seq_length}) exceeds recommended maximum "
                f"({self.max_supported_length}). Performance may degrade.")

        # Standard BERT attention calculations
        mixed_query_layer = self.query(hidden_states)

        # Handle cross-attention if encoder_hidden_states is provided
        is_cross_attention = encoder_hidden_states is not None
        if is_cross_attention:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        # Reshape for multi-head attention
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Calculate attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        # Apply ALiBi bias for self-attention (not for cross-attention)
        if not is_cross_attention:
            try:
                # Create position difference matrix dynamically
                position_ids = torch.arange(seq_length, device=attention_scores.device)
                distance = position_ids.unsqueeze(1) - position_ids.unsqueeze(0)  # [seq_len, seq_len]

                # Get absolute distance (since we want to penalize attention based on distance)
                # and negate (since we're adding this to attention scores before softmax)
                distance = -torch.abs(distance)  # [seq_len, seq_len]

                # Apply slopes to distances for each head
                # slopes: [num_heads], distance: [seq_len, seq_len]
                # -> alibi_bias: [num_heads, seq_len, seq_len]
                alibi_bias = distance.unsqueeze(0) * self.slopes.unsqueeze(1).unsqueeze(2)

                # Add bias to attention scores
                # attention_scores: [batch_size, num_heads, seq_len, seq_len]
                # alibi_bias: [num_heads, seq_len, seq_len]
                attention_scores = attention_scores + alibi_bias.unsqueeze(0)

            except Exception as e:
                logger.error(f"ALiBi application failed: {e}")
                # Continue without ALiBi rather than failing completely

        # Scale attention scores
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Apply attention mask
        if attention_mask is not None:
            # Apply the attention mask (precomputed for all layers in BertModel forward)
            attention_scores = attention_scores + attention_mask

        # Softmax normalization
        attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # Apply head mask
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # Apply attention to values
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        # Reshape back
        new_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_shape)

        # Outputs
        outputs = (context_layer,)
        if output_attentions:
            outputs = outputs + (attention_probs,)

        return outputs


def modify_bert_for_alibi(model):
    """
    Modify a standard BERT model to use ALiBi attention with improved safety checks.

    Args:
        model: A BertForMaskedLM model instance

    Returns:
        The modified model with ALiBi attention
    """
    # Verify the model config has necessary parameters
    if not hasattr(model.config, 'num_attention_heads'):
        raise ValueError("Model config missing 'num_attention_heads'")

    logger.info(f"Modifying BERT model to use ALiBi with {model.config.num_attention_heads} attention heads")

    # Set ALiBi-specific configurations
    model.config.use_alibi = True
    model.config.position_embedding_type = "alibi"

    # Replace standard self-attention with ALiBi self-attention in each layer
    for i, layer in enumerate(model.bert.encoder.layer):
        old_attention = layer.attention.self

        # Create new ALiBi attention
        new_attention = BertSelfAttentionWithALiBi(model.config)

        # Copy weights from old attention to new attention
        new_attention.query = old_attention.query
        new_attention.key = old_attention.key
        new_attention.value = old_attention.value
        new_attention.dropout = old_attention.dropout

        # Replace the attention layer
        layer.attention.self = new_attention

        logger.info(f"Replaced attention in layer {i} with ALiBi attention")
        model.resize_token_embeddings(len(model.get_input_embeddings().weight))

    logger.info("All layers successfully modified to use ALiBi attention")
    return model


def create_genomic_bert_config(
        vocab_size,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        use_alibi=True,
        max_supported_length=16384,
):
    """
    Create a BERT config for genomic data with ALiBi support.

    Args:
        vocab_size: Size of the vocabulary
        hidden_size: Size of the hidden layers
        num_hidden_layers: Number of hidden layers
        num_attention_heads: Number of attention heads
        intermediate_size: Size of the intermediate layers
        hidden_act: Activation function for hidden layers
        hidden_dropout_prob: Dropout probability for hidden layers
        attention_probs_dropout_prob: Dropout probability for attention
        max_position_embeddings: Maximum sequence length supported by position embeddings
        type_vocab_size: Size of the token type vocabulary
        initializer_range: Range for weight initialization
        layer_norm_eps: Epsilon for layer normalization
        pad_token_id: ID of the padding token
        use_alibi: Whether to use ALiBi for position representation
        max_supported_length: Maximum sequence length supported by the model

    Returns:
        BertConfig: Configuration for a genomic BERT model
    """
    from transformers import BertConfig

    config = BertConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        hidden_act=hidden_act,
        hidden_dropout_prob=hidden_dropout_prob,
        attention_probs_dropout_prob=attention_probs_dropout_prob,
        max_position_embeddings=max_position_embeddings,
        type_vocab_size=type_vocab_size,
        initializer_range=initializer_range,
        layer_norm_eps=layer_norm_eps,
        pad_token_id=pad_token_id,
    )

    # Add custom config for ALiBi
    config.use_alibi = use_alibi
    config.position_embedding_type = "alibi" if use_alibi else "absolute"

    # Add max supported length to config
    config.max_supported_length = max_supported_length

    return config


def create_genomic_bert_model(config):
    """
    Create a BERT model for genomic data with ALiBi that properly inherits from GenerationMixin.

    Args:
        config: BERT configuration

    Returns:
        BertForMaskedLM: BERT model with ALiBi attention and proper inheritance
    """
    from transformers import BertForMaskedLM
    from transformers.generation.utils import GenerationMixin

    # Define a custom class that inherits properly
    class GenomicBertForMaskedLM(BertForMaskedLM, GenerationMixin):
        pass

    logger.info(f"Initializing model with vocab_size: {config.vocab_size}")
    model = GenomicBertForMaskedLM(config)

    # Verify embedding size matches config vocab_size
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if embedding_size != config.vocab_size:
        logger.warning(f"Embedding size ({embedding_size}) doesn't match config vocab_size ({config.vocab_size})")
        logger.info(f"Resizing token embeddings to {config.vocab_size}")
        model.resize_token_embeddings(config.vocab_size)

    # Apply ALiBi if specified in config
    if config.use_alibi:
        model = modify_bert_for_alibi(model)

    return model