"""
Modified version of training/accelerate_utils.py that supports loading
pre-trained tokenizers instead of training them and implements SDPA
for efficient training.
"""

import os
import math
import logging
import copy
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import get_scheduler, BertForMaskedLM
from transformers.models.bert.modeling_bert import BertSelfAttention
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from accelerate import DistributedDataParallelKwargs

from tqdm import tqdm

from data.data_collator import GenomicDataset, GenomicMLMDataCollator
from modeling.alibi_attention import create_genomic_bert_config, create_genomic_bert_model, modify_bert_for_alibi
from training.train_utils import (
    generate_test_sequences,
    test_sequence_length_extrapolation,
    setup_mlm_data_collator
)

logger = logging.getLogger(__name__)


# SDPA Self-Attention Implementation
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
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=None,
            output_attentions=False,
    ):
        """
        Efficient attention implementation using SDPA with optional ALiBi bias.
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
            sdpa_attention_mask = attention_mask
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


def modify_bert_attention(model, attention_type, use_alibi=False):
    """
    Modify a BERT model to use the specified attention type.

    Args:
        model: A BertForMaskedLM model instance
        attention_type: Type of attention mechanism to use ("standard", "alibi", or "sdpa")
        use_alibi: Whether to use ALiBi bias with SDPA attention

    Returns:
        The modified model with the specified attention type
    """
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


def ensure_valid_batch_indices(batch, tokenizer, model, accelerator=None):
    """
    Ensure all token IDs in a batch are within the valid vocabulary range.
    Compatible with Accelerate's model wrapping.

    Args:
        batch: The batch of inputs
        tokenizer: The tokenizer
        model: The model (might be wrapped by Accelerator)
        accelerator: Optional accelerator instance for unwrapping the model

    Returns:
        Updated batch with validated indices
    """
    # If no input_ids, nothing to validate
    if 'input_ids' not in batch:
        return batch

    # Get the base model from any wrappers (like DistributedDataParallel)
    unwrapped_model = model

    # If we have an accelerator, use it to unwrap the model
    if accelerator is not None:
        unwrapped_model = accelerator.unwrap_model(model)
    # Fallback for DistributedDataParallel if no accelerator is available
    elif hasattr(model, 'module'):
        unwrapped_model = model.module

    # Get model's vocabulary size (embedding dimension)
    embedding_size = unwrapped_model.get_input_embeddings().weight.shape[0]

    # Check if any indices are out of bounds
    input_ids = batch['input_ids']
    max_id = input_ids.max().item()

    if max_id >= embedding_size:
        logger.warning(f"Found token ID {max_id} exceeding model vocab size {embedding_size}")

        # Create a mask of out-of-bounds IDs
        invalid_mask = input_ids >= embedding_size

        if invalid_mask.any():
            # Count invalid indices
            num_invalid = invalid_mask.sum().item()
            logger.warning(f"Clamping {num_invalid} token IDs to valid range")

            # Replace out-of-bounds IDs with unk_token_id
            unk_token_id = getattr(tokenizer, 'unk_token_id', 0) if tokenizer else 0
            input_ids = torch.where(invalid_mask,
                                    torch.tensor(unk_token_id, device=input_ids.device, dtype=input_ids.dtype),
                                    input_ids)

            # Update the batch
            batch['input_ids'] = input_ids

            # If labels exist, update them too to avoid loss on invalid indices
            if 'labels' in batch:
                batch['labels'] = torch.where(invalid_mask,
                                              torch.tensor(-100, device=batch['labels'].device,
                                                           dtype=batch['labels'].dtype),
                                              batch['labels'])

    return batch


def train_with_accelerate(args, accelerator):
    """
    Main training function using Accelerate with support for selectable attention mechanisms.
    Updated to use standard HuggingFace tokenizers without custom classes.

    Args:
        args: Command-line arguments
        accelerator: Accelerator instance
    """
    logger = logging.getLogger(__name__)

    # Set random seed for reproducibility
    if args.seed is not None:
        # Use deterministic per-process seeds
        global_seed = args.seed
        process_seed = global_seed + accelerator.process_index

        # Set all RNG states with process-specific seed
        import random
        import numpy as np
        random.seed(process_seed)
        np.random.seed(process_seed)
        torch.manual_seed(process_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(process_seed)

        logger.info(f"Process {accelerator.process_index} using seed {process_seed} (base: {global_seed})")

    # Create output directory
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # Wait for all processes to sync
    accelerator.wait_for_everyone()

    # Load pre-trained tokenizer - SIMPLIFIED to use standard tokenizers
    logger.info(f"Loading pre-trained tokenizer from {args.tokenizer_path}")
    try:
        # Use genomic utilities to load any tokenizer type
        from tokenization.genomic_utils import load_genomic_tokenizer, test_tokenizer_oov_handling

        tokenizer = load_genomic_tokenizer(args.tokenizer_path)

        # Test tokenizer OOV handling
        if accelerator.is_main_process:
            try:
                test_tokenizer_oov_handling(tokenizer)
            except Exception as e:
                logger.error(f"Tokenizer test failed: {e}")
                logger.error("This indicates potential problems with token ID handling")

    except Exception as e:
        raise ValueError(f"Failed to load tokenizer from {args.tokenizer_path}: {e}")

    # Setup model with the specified attention type
    if args.model_path and os.path.exists(args.model_path):
        logger.info(f"Loading existing model from {args.model_path}")
        model = BertForMaskedLM.from_pretrained(args.model_path)

        # Check if model has correct attention type
        model_attention_type = getattr(model.config, 'attention_type', 'standard')
        if model_attention_type != args.attention_type:
            logger.info(f"Changing attention type from {model_attention_type} to {args.attention_type}")
            model = modify_bert_attention(model, args.attention_type, use_alibi=args.use_alibi)
    else:
        # Try to use the SDPA-enabled config function if attention_type is sdpa
        if args.attention_type == "sdpa":
            logger.info(f"Initializing new BERT model with {args.attention_type} attention for genomic data")

            # Use the SDPA-enabled config function
            config = create_genomic_bert_config_with_sdpa(
                vocab_size=tokenizer.vocab_size,
                hidden_size=args.hidden_size,
                num_hidden_layers=args.num_hidden_layers,
                num_attention_heads=args.num_attention_heads,
                intermediate_size=args.hidden_size * 4,
                max_position_embeddings=args.pre_training_length,
                attention_type=args.attention_type,
                use_alibi=args.use_alibi,
                max_supported_length=args.max_supported_model_length,
            )
        else:
            # Fall back to the original config function for non-SDPA attention types
            config = create_genomic_bert_config(
                vocab_size=tokenizer.vocab_size,
                hidden_size=args.hidden_size,
                num_hidden_layers=args.num_hidden_layers,
                num_attention_heads=args.num_attention_heads,
                intermediate_size=args.hidden_size * 4,
                max_position_embeddings=args.pre_training_length,
                use_alibi=args.use_alibi,
                attention_type=args.attention_type,
                max_supported_length=args.max_supported_model_length,
            )

        model = create_genomic_bert_model(config)

        # Modify the model with the selected attention mechanism
        if args.attention_type != "standard":
            model = modify_bert_attention(model, args.attention_type, use_alibi=args.use_alibi)

    # CRITICAL FIX: Prepare model for DDP if using ALiBi attention
    model = prepare_alibi_model_for_ddp(model)

    # CRITICAL FIX: Verify and synchronize model-tokenizer compatibility
    model = verify_model_tokenizer_compatibility(model, tokenizer)

    use_reverse_complement = True
    if args.disable_reverse_complement:
        use_reverse_complement = False
    elif hasattr(args, 'use_reverse_complement'):
        use_reverse_complement = args.use_reverse_complement

    # Prepare dataset
    train_dataset = GenomicDataset(
        args.input_files,
        tokenizer,
        pre_training_length=args.pre_training_length,
        max_inference_length=args.max_inference_length,
        mlm_probability=args.mlm_probability,
        chunk_size=args.chunk_size,
        stride=args.stride,
        sample_long_sequences=args.sample_long_sequences,
        max_safe_sequence_length=args.max_safe_sequence_length,
        use_reverse_complement=use_reverse_complement,
    )

    data_collator = setup_mlm_data_collator(
        tokenizer=tokenizer,
        mlm_probability=args.mlm_probability,
        model=model,
        max_seq_length=args.pre_training_length  # Pass pre_training_length to collator
    )

    # Define worker initialization function for DataLoader
    def worker_init_fn(worker_id):
        # Each worker needs a different seed derived from process-specific seed
        process_seed = args.seed + accelerator.process_index * 1000
        worker_seed = process_seed + worker_id

        # Initialize all random number generators
        import random
        import numpy as np
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        # Note: No need to set CUDA seed in workers

        logger.debug(f"Worker {worker_id} on process {accelerator.process_index} using seed {worker_seed}")

    # Configure DataLoader with proper process and worker-specific seeding
    if accelerator.num_processes > 1:
        # Create a deterministic distributed sampler
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            shuffle=True,
            seed=args.seed,  # Use the global seed for shuffling
            drop_last=False
        )

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            collate_fn=data_collator,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            sampler=train_sampler,
            worker_init_fn=worker_init_fn
        )
    else:
        # Single-process mode
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            collate_fn=data_collator,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=worker_init_fn
        )

    # Prepare optimizer
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Calculate training steps
    num_update_steps_per_epoch, max_train_steps = calculate_training_steps(
        train_dataloader, args.gradient_accumulation_steps, args.epochs
    )

    # Setup learning rate scheduler
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=max_train_steps,
    )

    # Apply sequence length overrides for testing if specified
    if args.test_sequence_length is not None:
        logger.info(f"TEST MODE: Enforcing sequence length of {args.test_sequence_length}")
        args.pre_training_length = args.test_sequence_length
        args.max_supported_model_length = args.test_sequence_length
        model.config.max_supported_length = min(getattr(model.config, "max_supported_length", 4096), 4096)

    # Before preparing with accelerator, store the distributed state
    is_distributed = accelerator.num_processes > 1
    has_sampler = is_distributed  # True if we're using a DistributedSampler

    # Prepare with accelerator
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    # IMPORTANT: Generate test sequences for extrapolation testing with synchronized seed
    # We do this after accelerator.prepare but before tests to ensure consistency
    test_lengths = [
        args.pre_training_length,
        args.pre_training_length * 2,
        args.pre_training_length * 4,
    ]
    if args.max_inference_length:
        test_lengths.append(args.max_inference_length)

    # For the extrapolation test, we'll use a fixed seed to generate the same
    # sequences on all processes to avoid synchronization issues during testing
    extrapolation_test_seed = 12345  # A fixed seed just for generating test sequences

    # Only generate the test sequences on the main process
    if accelerator.is_main_process:
        # Temporarily set a fixed seed
        saved_state = random.getstate()
        random.seed(extrapolation_test_seed)
        extrapolation_test_seqs = generate_test_sequences(test_lengths)
        random.setstate(saved_state)  # Restore original random state

        # Test initial extrapolation (only on main process)
        logger.info("\nInitial length extrapolation test:")
        test_sequence_length_extrapolation(accelerator, model, tokenizer, extrapolation_test_seqs)

    # Sync before continuing
    accelerator.wait_for_everyone()

    # Resume from checkpoint if needed
    starting_epoch, completed_steps = resume_from_checkpoint(accelerator, args, num_update_steps_per_epoch)

    # Compute total training batch size
    total_batch_size = args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    # Log training information
    logger.info("***** Running training *****")
    logger.info(f"  Attention type = {args.attention_type}")
    if args.attention_type == "sdpa":
        logger.info(f"  Using ALiBi with SDPA = {args.use_alibi}")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num epochs = {args.epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    logger.info(f"  Using distributed sampler: {has_sampler}")
    logger.info(f"  RNG strategy: Process-specific seeds with manual control")
    logger.info(f"  Global seed: {args.seed}")

    # Setup progress bar (only on main process)
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_main_process)
    progress_bar.update(completed_steps)

    # Main training loop
    for epoch in range(starting_epoch, args.epochs):
        model.train()
        total_loss = 0
        valid_steps = 0  # Count successful steps for averaging

        # Set epoch for distributed sampler - crucially important for proper shuffling
        if has_sampler and hasattr(train_dataloader, "sampler") and hasattr(train_dataloader.sampler, "set_epoch"):
            train_dataloader.sampler.set_epoch(epoch)
            logger.info(f"Set epoch {epoch} for distributed sampler")

        # Process batches with improved error handling
        for step, batch in enumerate(train_dataloader):
            # Process batch safely
            loss = safe_training_step(model, batch, accelerator, args)

            if loss is not None:
                # Successfully processed batch
                total_loss += loss
                valid_steps += 1

                # Update on accumulation steps
                if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    # Clip gradients
                    if args.max_grad_norm is not None:
                        accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                    # Optimizer and scheduler step
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                    # Update progress
                    progress_bar.update(1)
                    completed_steps += 1

                    # Log loss periodically
                    if completed_steps % args.logging_steps == 0 and valid_steps > 0:
                        avg_loss = total_loss.item() / valid_steps
                        logger.info(f"Epoch: {epoch}, Step: {completed_steps}, Loss: {avg_loss:.4f}")
                        total_loss = 0
                        valid_steps = 0

                    # Save checkpoint if needed
                    if args.checkpointing_steps is not None and completed_steps % args.checkpointing_steps == 0:
                        save_checkpoint(
                            accelerator, args, epoch, completed_steps,
                            model, optimizer, lr_scheduler, tokenizer
                        )
            else:
                # Error in this batch - skip and continue
                logger.warning(f"Skipping problematic batch at Epoch {epoch}, Step {step}")
                # Explicit synchronization
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()

        # Save checkpoint after each epoch
        save_checkpoint(
            accelerator, args, epoch, completed_steps,
            model, optimizer, lr_scheduler, tokenizer
        )

    try:
        if accelerator.is_main_process:
            final_dir = os.path.join(args.output_dir, "final")
            logger.info(f"Saving final model to {final_dir}")
            save_model(accelerator, model, tokenizer, final_dir)

            # Create a standalone copy of the model for testing
            logger.info("\nPreparing model for final length extrapolation test...")
            test_model = accelerator.unwrap_model(model)

            # Create a fresh copy to fully detach from distributed state
            test_model = copy.deepcopy(test_model)

            # Keep model on GPU for final test (just like initial test)
            # Get the current GPU device
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            test_model = test_model.to(device)
            test_model.eval()

            # Create a GPU accelerator reference similar to the original one
            class SimpleGPUAccelerator:
                def __init__(self, device):
                    self.device = device

                def unwrap_model(self, model):
                    return model

            gpu_accelerator = SimpleGPUAccelerator(device)

            # Final length extrapolation test
            logger.info("\nFinal length extrapolation test:")

            # Use the same sequences we generated earlier for consistent testing
            # This avoids RNG synchronization issues
            saved_state = random.getstate()
            random.seed(extrapolation_test_seed)
            test_seqs = generate_test_sequences(test_lengths)
            random.setstate(saved_state)

            test_sequence_length_extrapolation(gpu_accelerator, test_model, tokenizer, test_seqs)
    except Exception as e:
        logger.error(f"Error during final model saving or testing: {e}")
        logger.exception("Detailed traceback:")  # Print full traceback for debugging
        if accelerator.is_main_process:
            logger.info("Training completed but final model saving failed. Use the last checkpoint instead.")

    # At the end of your train_with_accelerate function
    accelerator.wait_for_everyone()
    # Let Accelerate handle cleanup operations
    if accelerator.is_main_process:
        logger.info("Training completed, performing final cleanup")

def prepare_alibi_model_for_ddp(model):
    """
    Prepare an ALiBi model for DDP by handling unused position embeddings.
    When using ALiBi attention, position embeddings aren't used in forward pass
    but still exist in the model, causing DDP to complain about unused parameters.

    Args:
        model: The BERT model with ALiBi attention

    Returns:
        The prepared model
    """
    logger.info("Preparing ALiBi model for distributed training")

    # Check if model uses ALiBi attention
    is_alibi = getattr(model.config, 'use_alibi', False) or getattr(model.config, 'attention_type',
                                                                    '') == 'alibi' or getattr(model.config,
                                                                                              'attention_type',
                                                                                              '') == 'sdpa'

    if not is_alibi:
        logger.info("Model does not use ALiBi attention, no special preparation needed")
        return model

    # Handle position embeddings
    try:
        # Check if position embeddings exist
        if hasattr(model.bert.embeddings, 'position_embeddings'):
            logger.info("Handling position embeddings for ALiBi model")

            # Freeze position embeddings to prevent gradient issues in DDP
            model.bert.embeddings.position_embeddings.requires_grad = False
            logger.info("Froze position embeddings for ALiBi model")

            # Set initialization flag in config to track this change
            model.config.position_embeddings_frozen = True
    except Exception as e:
        logger.warning(f"Error preparing ALiBi model for DDP: {e}")

    return model


def setup_accelerator(args):
    """
    Set up Accelerator with proper configuration for distributed training.
    Compatible with the latest version of Accelerate library.

    Args:
        args: Command-line arguments

    Returns:
        Accelerator instance
    """

    if args.gpu_ids and "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    # Create DDP kwargs configuration
    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=True,
        static_graph=False
    )

    # Create checkpoint configuration
    project_config = None
    if args.output_dir:
        project_config = ProjectConfiguration(
            project_dir=args.output_dir,
            logging_dir=os.path.join(args.output_dir, "logs")
        )

    # Set up device configuration with explicit device IDs
    device_placement_config = None
    if torch.cuda.is_available():
        device_placement_config = {"device_map": "auto"}

    # IMPORTANT: Disable RNG synchronization to prevent collective mismatches
    # We'll handle RNG manually instead
    accelerator_config = {
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "project_config": project_config,
        "device_placement": device_placement_config,
        "kwargs_handlers": [ddp_kwargs],  # Use kwargs_handlers for latest Accelerate
    }

    if hasattr(args, 'log_with_tensorboard') and args.log_with_tensorboard:
        accelerator_config["log_with"] = "tensorboard"

    # Create accelerator with disabled RNG synchronization
    accelerator = Accelerator(**accelerator_config)

    # Report accelerator configuration
    logger.info(f"Accelerator configuration:")
    logger.info(f"  Distributed type: {accelerator.distributed_type}")
    logger.info(f"  Num processes: {accelerator.num_processes}")
    logger.info(f"  Process index: {accelerator.process_index}")
    logger.info(f"  Device: {accelerator.device}")
    logger.info(f"  Mixed precision: {accelerator.mixed_precision}")
    logger.info(f"  DDP kwargs: find_unused_parameters=True, static_graph=False")
    logger.info(f"  RNG synchronization: manually controlled")

    # Check SDPA support
    if torch.cuda.is_available():
        cap_major = torch.cuda.get_device_capability(0)[0]
        logger.info(f"  CUDA Capability: {cap_major}.x")
        if cap_major >= 8:
            logger.info("  Hardware-accelerated SDPA is supported (Ampere architecture or newer)")
        else:
            logger.info("  Hardware-accelerated SDPA may not be fully optimized (pre-Ampere architecture)")
    else:
        logger.info("  SDPA will use CPU implementation")

    return accelerator

def calculate_training_steps(train_dataloader, gradient_accumulation_steps, num_epochs):
    """Calculate the number of update steps for training."""
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    max_train_steps = num_epochs * num_update_steps_per_epoch
    return num_update_steps_per_epoch, max_train_steps


def save_checkpoint(accelerator, args, epoch, step, model, optimizer, lr_scheduler, tokenizer):
    """Save a checkpoint with proper state management."""
    # Determine checkpoint path (epoch or step-based)
    if step % args.save_steps == 0:
        checkpoint_path = os.path.join(args.output_dir, f"step_{step}")
    else:
        checkpoint_path = os.path.join(args.output_dir, f"epoch_{epoch}")

    logger.info(f"Saving checkpoint to {checkpoint_path}")

    # Use Accelerate's state saving
    accelerator.save_state(checkpoint_path)

    # Save config.json on main process only
    if accelerator.is_main_process:
        # Unwrap the model to access its config
        unwrapped_model = accelerator.unwrap_model(model)

        # Save just the config.json file (not model weights again)
        unwrapped_model.config.save_pretrained(checkpoint_path)
        logger.info(f"Saved model config to {checkpoint_path}")

    # Save tokenizer on main process only
    if accelerator.is_main_process and tokenizer is not None:
        tokenizer.save_pretrained(checkpoint_path)

    # Clean up old checkpoints (only keep N most recent)
    if accelerator.is_main_process and args.save_total_limit is not None:
        cleanup_checkpoints(args.output_dir, args.save_total_limit)

    # Wait for checkpoint saving to complete
    accelerator.wait_for_everyone()

def cleanup_checkpoints(output_dir, save_total_limit):
    """Clean up old checkpoints, keeping only the most recent ones."""
    import os
    import shutil
    import re

    # Get all checkpoint directories
    checkpoint_dirs = []
    pattern = re.compile(r"(epoch|step)_(\d+)")

    for dirname in os.listdir(output_dir):
        if not os.path.isdir(os.path.join(output_dir, dirname)):
            continue

        match = pattern.match(dirname)
        if match:
            checkpoint_type = match.group(1)  # 'epoch' or 'step'
            checkpoint_num = int(match.group(2))
            checkpoint_dirs.append((dirname, checkpoint_type, checkpoint_num))

    # Sort by type and number
    checkpoint_dirs.sort(key=lambda x: (x[1], x[2]))

    # Keep only the most recent ones
    if len(checkpoint_dirs) > save_total_limit:
        dirs_to_remove = checkpoint_dirs[:-save_total_limit]
        for dirname, _, _ in dirs_to_remove:
            logger.info(f"Removing old checkpoint: {dirname}")
            shutil.rmtree(os.path.join(output_dir, dirname))


def resume_from_checkpoint(accelerator, args, num_update_steps_per_epoch):
    """Resume training from a checkpoint with proper state tracking."""
    starting_epoch = 0
    completed_steps = 0

    if not args.resume_from_checkpoint:
        return starting_epoch, completed_steps

    checkpoint_path = args.resume_from_checkpoint

    # Find latest checkpoint if requested
    if args.resume_from_checkpoint == "latest":
        # Look in both epoch-based and step-based directories
        checkpoint_dirs = []

        # Check epoch directories
        epoch_dirs = [d for d in os.listdir(args.output_dir)
                      if d.startswith("epoch_") and os.path.isdir(os.path.join(args.output_dir, d))]
        checkpoint_dirs.extend([(d, int(d.split("_")[-1])) for d in epoch_dirs])

        # Check step directories
        step_dirs = [d for d in os.listdir(args.output_dir)
                     if d.startswith("step_") and os.path.isdir(os.path.join(args.output_dir, d))]
        checkpoint_dirs.extend([(d, int(d.split("_")[-1])) for d in step_dirs])

        if not checkpoint_dirs:
            logger.info("No checkpoints found, starting training from scratch")
            return starting_epoch, completed_steps

        # Sort by number (either epoch or step)
        checkpoint_dirs.sort(key=lambda x: x[1])
        latest_dir = checkpoint_dirs[-1][0]
        checkpoint_path = os.path.join(args.output_dir, latest_dir)

    # Log checkpoint info
    logger.info(f"Resuming from checkpoint: {checkpoint_path}")

    # Load checkpoint state with proper RNG handling
    try:
        # First try loading with RNG states
        accelerator.load_state(checkpoint_path)
        logger.info("Successfully loaded checkpoint including RNG states")
    except RuntimeError as e:
        if "RNG state" in str(e):
            logger.warning(f"Failed to load RNG states from checkpoint: {e}")
            logger.warning("Trying to load checkpoint without RNG states")
            # Try loading without RNG states
            accelerator.load_state(checkpoint_path, load_rng_state=False)
            logger.info("Successfully loaded checkpoint without RNG states")

            # Reset RNG with process-specific seeds
            process_seed = args.seed + accelerator.process_index * 100
            import random
            import numpy as np
            random.seed(process_seed)
            np.random.seed(process_seed)
            torch.manual_seed(process_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(process_seed)
        else:
            # Re-raise if it's not related to RNG states
            raise

    # Extract epoch and step from checkpoint path
    path = Path(checkpoint_path)
    if "epoch_" in path.name:
        starting_epoch = int(path.name.split("_")[-1])
        completed_steps = starting_epoch * num_update_steps_per_epoch
    elif "step_" in path.name:
        completed_steps = int(path.name.split("_")[-1])
        starting_epoch = completed_steps // num_update_steps_per_epoch

    logger.info(f"Resuming from epoch {starting_epoch}, step {completed_steps}")

    # Force synchronize CUDA to ensure clean resumption
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    return starting_epoch, completed_steps

def save_model(accelerator, model, tokenizer, output_dir):
    """Save the model and tokenizer using Accelerate with improved robustness for distributed training."""
    # Create output directory if needed
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)

        # Unwrap model and save on main process
        unwrapped_model = accelerator.unwrap_model(model)

        # Temporarily remove tokenizer from config if present
        tokenizer_from_config = None
        if hasattr(unwrapped_model.config, 'tokenizer'):
            tokenizer_from_config = unwrapped_model.config.tokenizer
            delattr(unwrapped_model.config, 'tokenizer')

        try:
            # Save model and config using HuggingFace's built-in methods
            unwrapped_model.save_pretrained(output_dir)
            logger.info(f"Model saved to {output_dir}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
        finally:
            # Restore tokenizer in config
            if tokenizer_from_config is not None:
                unwrapped_model.config.tokenizer = tokenizer_from_config

        # Save tokenizer if provided
        if tokenizer is not None:
            try:
                tokenizer.save_pretrained(output_dir)
                logger.info(f"Tokenizer saved to {output_dir}")
            except Exception as e:
                logger.error(f"Error saving tokenizer: {e}")

    # No need to wait for other processes - this was causing the timeout
    # Removed: accelerator.wait_for_everyone()


def verify_model_tokenizer_compatibility(model, tokenizer):
    """Verify and ensure the model and tokenizer are compatible."""
    # First, fix any token ID issues in the tokenizer
    if hasattr(tokenizer, 'mask_token_id') and tokenizer.mask_token_id is not None:
        if tokenizer.mask_token_id >= tokenizer.vocab_size:
            tokenizer.mask_token_id = 1  # Use consistent safe ID
            logger.warning(f"Fixed tokenizer mask_token_id to use ID 1")

    # Then, add this to model config
    model.config.mask_token_id = tokenizer.mask_token_id

    # Ensure embedding size matches
    new_size = tokenizer.vocab_size
    model.resize_token_embeddings(new_size)

    # Verify the change was successful
    if model.get_input_embeddings().weight.shape[0] != new_size:
        logger.error(f"Failed to resize model embeddings to {new_size}")

    return model


def safe_training_step(model, batch, accelerator, args):
    """Execute a single training step with robust error handling."""
    import torch
    import gc

    try:
        # Validate token IDs before forward pass - pass the accelerator
        tokenizer = getattr(accelerator.unwrap_model(model).config, 'tokenizer', None)
        batch = ensure_valid_batch_indices(batch, tokenizer, model, accelerator)

        # Check available GPU memory
        if torch.cuda.is_available():
            device = next(model.parameters()).device
            available_memory = torch.cuda.get_device_properties(device).total_memory - torch.cuda.memory_allocated(
                device)
            available_gb = available_memory / (1024 ** 3)
            logger.debug(f"Available GPU memory: {available_gb:.2f} GB")

        # Get batch dimensions
        if 'input_ids' in batch:
            batch_size, seq_length = batch['input_ids'].shape

            # Implement dynamic handling based on sequence length
            # Longer sequences need more memory - adjust batch dynamically if needed
            if seq_length > 2048:
                # For very long sequences, split batch if memory is tight
                if torch.cuda.is_available() and available_gb < 2.0:  # Conservative threshold
                    logger.warning(
                        f"Sequence length {seq_length} with low memory ({available_gb:.2f} GB). Processing in micro-batches.")
                    return _process_in_micro_batches(model, batch, accelerator, args)

        # Forward pass
        outputs = model(**batch)
        loss = outputs.loss / args.gradient_accumulation_steps

        # Backward pass with gradient scaling if FP16 is enabled
        accelerator.backward(loss)

        return loss.detach().float()

    except RuntimeError as e:
        error_msg = str(e)

        # Handle different error types
        if "CUDA out of memory" in error_msg:
            logger.error(f"CUDA OOM error during training: {e}")

            # Try to recover by processing in micro-batches
            if 'input_ids' in batch and batch['input_ids'].shape[0] > 1:
                logger.info("Attempting recovery by processing in micro-batches")
                try:
                    return _process_in_micro_batches(model, batch, accelerator, args)
                except Exception as micro_error:
                    logger.error(f"Micro-batch recovery failed: {micro_error}")

            # Free memory
            torch.cuda.empty_cache()
            gc.collect()

        elif "index out of bounds" in error_msg or "indexSelectLargeIndex" in error_msg:
            logger.error(f"CUDA indexing error: {e}")
            # Log input shapes to help debug
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    logger.error(f"  {k} shape: {v.shape}, min: {v.min().item()}, max: {v.max().item()}")
        else:
            logger.error(f"Unexpected error: {e}")

        # Force synchronize
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        return None


def _process_in_micro_batches(model, batch, accelerator, args, micro_batch_size=1):
    """Process a batch in smaller micro-batches to handle memory constraints."""
    import torch

    # Get the original batch size
    full_batch_size = batch['input_ids'].shape[0]

    # If batch size is already 1, we can't split further
    if full_batch_size <= micro_batch_size:
        raise ValueError("Batch size is already at minimum, cannot process in micro-batches")

    # Initialize accumulated loss using accelerator's device
    accumulated_loss = torch.tensor(0.0, device=accelerator.device)

    # Process each micro-batch
    for i in range(0, full_batch_size, micro_batch_size):
        # Get micro-batch indices
        end_idx = min(i + micro_batch_size, full_batch_size)
        logger.info(
            f"Processing micro-batch {i // micro_batch_size + 1}/{(full_batch_size + micro_batch_size - 1) // micro_batch_size}")

        # Create micro-batch
        micro_batch = {k: v[i:end_idx] for k, v in batch.items()}

        # Validate token IDs in micro-batch - pass the accelerator
        tokenizer = getattr(accelerator.unwrap_model(model).config, 'tokenizer', None)
        micro_batch = ensure_valid_batch_indices(micro_batch, tokenizer, model, accelerator)

        # Forward pass
        outputs = model(**micro_batch)
        micro_loss = outputs.loss / args.gradient_accumulation_steps / (full_batch_size / micro_batch_size)

        # Backward pass - use accelerator
        accelerator.backward(micro_loss)

        # Add to accumulated loss
        accumulated_loss += micro_loss.detach() * (end_idx - i)

    # Return average loss
    return accumulated_loss / full_batch_size