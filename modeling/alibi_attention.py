"""
Modified version of modeling/alibi_attention.py with pickling support.

Move the GenomicBertForMaskedLM class outside of the create_genomic_bert_model
function to fix multiprocessing pickling issues.
"""

import math
import torch
import logging
from typing import Optional, Tuple, Union

from transformers.models.bert.modeling_bert import BertSelfAttention
from transformers import BertForMaskedLM
from transformers.generation.utils import GenerationMixin
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class BertSelfAttentionWithSDPA(BertSelfAttention):
    """
    BERT self-attention using PyTorch's Scaled Dot-Product Attention (SDPA).
    This implementation provides performance benefits on modern GPUs while
    supporting variable sequence lengths.
    """

    def __init__(self, config, position_embedding_type=None):
        super().__init__(config, position_embedding_type)

        # Set default attention implementation
        self.sdpa_implementation = "sdpa"
        # This attribute is used for ALiBi
        self.use_alibi = getattr(config, "use_alibi", False)

        # Get maximum supported length from config or use a reasonable default
        self.max_supported_length = getattr(config, "max_supported_length", 4096)

        # Pre-compute slopes for ALiBi if needed
        if self.use_alibi:
            # Pre-compute slopes for each attention head - follows geometric sequence
            num_heads = self.num_attention_heads
            m = torch.arange(1, num_heads + 1, dtype=torch.float32)
            m = m * 8 / num_heads  # Scale factor based on the ALiBi paper
            self.register_buffer("slopes", 1.0 / (2 ** m))

        # Disable position embeddings since SDPA with ALiBi replaces them
        if self.use_alibi:
            self.position_embedding_type = None

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
        Efficient attention implementation using SDPA with optional ALiBi bias.

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
        query_layer = self.transpose_for_scores(mixed_query_layer)  # [batch, heads, seq_len, head_dim]
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Calculate ALiBi bias if needed
        alibi_bias = None
        if self.use_alibi and not is_cross_attention:
            try:
                # Create position difference matrix dynamically
                position_ids = torch.arange(seq_length, device=hidden_states.device)
                distance = position_ids.unsqueeze(1) - position_ids.unsqueeze(0)  # [seq_len, seq_len]

                # Get absolute distance and negate
                distance = -torch.abs(distance)  # [seq_len, seq_len]

                # Apply slopes to distances for each head
                # slopes: [num_heads], distance: [seq_len, seq_len]
                # -> alibi_bias: [num_heads, seq_len, seq_len]
                alibi_bias = distance.unsqueeze(0) * self.slopes.unsqueeze(1).unsqueeze(2)

                # Expand alibi_bias to [batch_size, num_heads, seq_len, seq_len]
                alibi_bias = alibi_bias.unsqueeze(0).expand(batch_size, -1, -1, -1)
            except Exception as e:
                logger.error(f"ALiBi bias calculation failed: {e}")

        # Process attention mask for SDPA
        # SDPA expects a float mask with -inf for masked positions, but regular transformers might use additive mask
        if attention_mask is not None:
            # Convert additive mask to causal mask format for SDPA
            # Transformers uses 0/-10000 for visible/masked, SDPA expects 0/-inf for masked/visible
            sdpa_attention_mask = (1.0 - attention_mask[:, None, None, :]) * -10000.0
        else:
            sdpa_attention_mask = None

        # Check if alibi_bias needs to be combined with attention mask
        if alibi_bias is not None and sdpa_attention_mask is not None:
            # Add ALiBi bias to attention mask
            sdpa_attention_mask = sdpa_attention_mask + alibi_bias
        elif alibi_bias is not None:
            # Use ALiBi bias as attention mask
            sdpa_attention_mask = alibi_bias

        # Use PyTorch's SDPA
        try:
            # Scale query for better numerical stability (also handled internally by F.sdpa)
            # Note: we don't need to divide by sqrt(head_dim) as it's done inside F.scaled_dot_product_attention
            # Use SDPA with our masks

            attn_weights = None
            if output_attentions:
                # Manual computation for getting attention weights
                attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
                attention_scores = attention_scores / math.sqrt(self.attention_head_size)

                if sdpa_attention_mask is not None:
                    attention_scores = attention_scores + sdpa_attention_mask

                # Normalize attention scores to probabilities
                attn_weights = torch.nn.functional.softmax(attention_scores, dim=-1)

            # Apply head mask if needed
            if head_mask is not None:
                if output_attentions:
                    attn_weights = attn_weights * head_mask.unsqueeze(1).unsqueeze(-1)
                query_layer = query_layer * head_mask.unsqueeze(1).unsqueeze(-1)

            # Apply dropout (handled inside F.scaled_dot_product_attention)
            # Note: If we use SDPA with custom scaling and bias, we might need to apply dropout manually

            # Call scaled_dot_product_attention
            context_layer = F.scaled_dot_product_attention(
                query_layer,
                key_layer,
                value_layer,
                attn_mask=sdpa_attention_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=False  # We handle our own masking above
            )

        except Exception as e:
            logger.error(f"SDPA failed: {e}. Falling back to standard attention.")

            # Calculate attention scores in the standard way
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

            # Apply ALiBi bias if needed
            if alibi_bias is not None:
                attention_scores = attention_scores + alibi_bias

            # Scale attention scores
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)

            # Apply attention mask
            if attention_mask is not None:
                attention_scores = attention_scores + attention_mask

            # Softmax normalization
            attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)
            attention_probs = self.dropout(attention_probs)

            # Apply head mask
            if head_mask is not None:
                attention_probs = attention_probs * head_mask.unsqueeze(1).unsqueeze(-1)

            # Apply attention to values
            context_layer = torch.matmul(attention_probs, value_layer)

            # Save attention weights if needed
            if output_attentions:
                attn_weights = attention_probs

        # Reshape back
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_shape)

        # Outputs
        outputs = (context_layer,)
        if output_attentions:
            outputs = outputs + (attn_weights,)

        return outputs


def modify_bert_for_sdpa(model):
    """
    Modify a standard BERT model to use SDPA attention, optionally with ALiBi.

    Args:
        model: A BertForMaskedLM model instance

    Returns:
        The modified model with SDPA attention
    """
    # Verify the model config has necessary parameters
    if not hasattr(model.config, 'num_attention_heads'):
        raise ValueError("Model config missing 'num_attention_heads'")

    # Extract whether to use ALiBi from model config
    use_alibi = getattr(model.config, 'use_alibi', False)
    logger.info(
        f"Modifying BERT model to use SDPA attention with {model.config.num_attention_heads} attention heads (ALiBi: {use_alibi})")

    # Set SDPA-specific configurations
    model.config.use_sdpa = True
    if use_alibi:
        model.config.position_embedding_type = "alibi"
    else:
        model.config.position_embedding_type = "absolute"  # Keep standard position embeddings if not using ALiBi

    # Replace standard self-attention with SDPA self-attention in each layer
    for i, layer in enumerate(model.bert.encoder.layer):
        old_attention = layer.attention.self

        # Create new SDPA attention
        new_attention = BertSelfAttentionWithSDPA(model.config)

        # Copy weights from old attention to new attention
        new_attention.query = old_attention.query
        new_attention.key = old_attention.key
        new_attention.value = old_attention.value
        new_attention.dropout = old_attention.dropout

        # Replace the attention layer
        layer.attention.self = new_attention

        logger.info(f"Replaced attention in layer {i} with SDPA attention")

    # Resize token embeddings to ensure consistency
    model.resize_token_embeddings(len(model.get_input_embeddings().weight))

    logger.info("All layers successfully modified to use SDPA attention")
    return model


# Update the create_genomic_bert_config to support SDPA option
def create_genomic_bert_config_with_sdpa(
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
        attention_type="sdpa",  # Add SDPA as default option
        use_alibi=True,
        max_supported_length=16384,
):
    """
    Create a BERT config for genomic data with SDPA support.

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
        attention_type: Type of attention mechanism ("standard", "alibi", or "sdpa")
        use_alibi: Whether to use ALiBi for position representation when using SDPA
        max_supported_length: Maximum sequence length supported by the model

    Returns:
        BertConfig: Configuration for a genomic BERT model
    """
    from transformers import BertConfig

    # Update use_alibi based on attention_type
    if attention_type == "alibi":
        use_alibi = True
    elif attention_type == "standard":
        use_alibi = False
    # For SDPA, use the passed use_alibi parameter

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

    # Add custom config for attention type
    config.use_alibi = use_alibi
    config.position_embedding_type = "alibi" if use_alibi else "absolute"
    config.attention_type = attention_type
    config.use_sdpa = (attention_type == "sdpa")

    # Add max supported length to config
    config.max_supported_length = max_supported_length

    return config


# Update the modify_bert_attention function to handle SDPA
def modify_bert_attention_with_sdpa(model, attention_type):
    """
    Modify a BERT model to use the specified attention type.

    Args:
        model: A BertForMaskedLM model instance
        attention_type: Type of attention mechanism to use ("standard", "alibi", or "sdpa")

    Returns:
        The modified model with the specified attention type
    """
    from transformers.models.bert.modeling_bert import BertSelfAttention
    from modeling.alibi_attention import modify_bert_for_alibi

    # Get current attention type (if set)
    current_attention_type = getattr(model.config, 'attention_type', 'standard')

    # If attention type is already set correctly, no need to modify
    if current_attention_type == attention_type:
        logger.info(f"Model already uses {attention_type} attention")
        return model

    # Handle different attention types
    if attention_type == "sdpa":
        logger.info("Modifying model to use SDPA attention")
        return modify_bert_for_sdpa(model)
    elif attention_type == "alibi":
        logger.info("Modifying model to use ALiBi attention")
        return modify_bert_for_alibi(model)
    elif attention_type == "standard":
        # Revert to standard attention if currently using ALiBi or SDPA
        if getattr(model.config, 'use_alibi', False) or getattr(model.config, 'use_sdpa',
                                                                False) or current_attention_type != 'standard':
            logger.info("Reverting model to use standard attention")

            # Reset any special attention configurations
            model.config.use_alibi = False
            model.config.use_sdpa = False
            model.config.position_embedding_type = "absolute"
            model.config.attention_type = "standard"

            # Replace special attention with standard BertSelfAttention in each layer
            for i, layer in enumerate(model.bert.encoder.layer):
                old_attention = layer.attention.self

                # Create new standard attention
                new_attention = BertSelfAttention(model.config)

                # Copy weights from old attention to new attention
                new_attention.query = old_attention.query
                new_attention.key = old_attention.key
                new_attention.value = old_attention.value
                new_attention.dropout = old_attention.dropout

                # Replace the attention layer
                layer.attention.self = new_attention

                logger.info(f"Replaced attention in layer {i} with standard attention")

            # Resize token embeddings to ensure compatibility
            model.resize_token_embeddings(len(model.get_input_embeddings().weight))

            logger.info("All layers successfully modified to use standard attention")
        else:
            logger.info("Model already uses standard attention")

        return model
    else:
        raise ValueError(f"Unsupported attention type: {attention_type}")

# Move the class to module level so it can be pickled properly
class GenomicBertForMaskedLM(BertForMaskedLM, GenerationMixin):
    """
    BERT for genomic data with proper inheritance from GenerationMixin.
    This class must be at module level (not inside a function) to support pickling
    for multiprocessing with DataLoader workers.
    """
    pass


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

    # Resize token embeddings to ensure consistency
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
        attention_type=None,  # Added parameter for consistency
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
        attention_type: Type of attention mechanism to use ("standard" or "alibi")
        max_supported_length: Maximum sequence length supported by the model

    Returns:
        BertConfig: Configuration for a genomic BERT model
    """
    from transformers import BertConfig

    # If attention_type is provided, use it to determine use_alibi
    if attention_type is not None:
        use_alibi = (attention_type == "alibi")

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
    config.attention_type = "alibi" if use_alibi else "standard"

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
    # Use the module-level class instead of defining it locally
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


# Definition of modify_bert_attention function
def modify_bert_attention(model, attention_type):
    """
    Modify a BERT model to use the specified attention type.

    Args:
        model: A BertForMaskedLM model instance
        attention_type: Type of attention mechanism to use ("standard" or "alibi")

    Returns:
        The modified model with the specified attention type
    """
    # Get current attention type (if set)
    current_attention_type = getattr(model.config, 'attention_type', 'standard')

    # If attention type is already set correctly, no need to modify
    if current_attention_type == attention_type:
        logger.info(f"Model already uses {attention_type} attention")
        return model

    if attention_type == "alibi":
        # Modify model to use ALiBi attention
        logger.info("Modifying model to use ALiBi attention")
        return modify_bert_for_alibi(model)
    elif attention_type == "standard":
        # If model is currently using ALiBi, revert to standard attention
        if getattr(model.config, 'use_alibi', False) or current_attention_type == 'alibi':
            logger.info("Reverting model to use standard attention")
            # We need to replace ALiBi attention with standard attention
            from transformers.models.bert.modeling_bert import BertSelfAttention

            # Reset any ALiBi-specific configurations
            model.config.use_alibi = False
            model.config.position_embedding_type = "absolute"
            model.config.attention_type = "standard"

            # Replace ALiBi attention with standard BertSelfAttention in each layer
            for i, layer in enumerate(model.bert.encoder.layer):
                old_attention = layer.attention.self

                # Create new standard attention
                new_attention = BertSelfAttention(model.config)

                # Copy weights from old attention to new attention
                new_attention.query = old_attention.query
                new_attention.key = old_attention.key
                new_attention.value = old_attention.value
                new_attention.dropout = old_attention.dropout

                # Replace the attention layer
                layer.attention.self = new_attention

                logger.info(f"Replaced ALiBi attention in layer {i} with standard attention")

            # Resize token embeddings to ensure compatibility
            model.resize_token_embeddings(len(model.get_input_embeddings().weight))

            logger.info("All layers successfully modified to use standard attention")
        else:
            logger.info("Model already uses standard attention")

        return model
    else:
        raise ValueError(f"Unsupported attention type: {attention_type}")