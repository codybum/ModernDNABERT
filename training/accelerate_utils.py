"""
Training utilities for genomic BERT model using PyTorch Accelerate.

This module provides the core training functionality, including setup of the Accelerate
environment, training loops, and checkpoint management.
"""

import os
import math
import logging
import gc
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
from torch.utils.data import DataLoader
from transformers import get_scheduler
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from tqdm import tqdm

from data.data_collator import GenomicDataset, GenomicMLMDataCollator
from modeling.alibi_attention import create_genomic_bert_config, modify_bert_for_alibi, create_genomic_bert_model
from tokenization.genomic_tokenizer import setup_tokenizer
from training.train_utils import generate_test_sequences, test_sequence_length_extrapolation

logger = logging.getLogger(__name__)


def setup_accelerator(args):
    """
    Set up an Accelerate environment with proper configuration.

    Args:
        args: Command-line arguments

    Returns:
        Accelerator: Configured accelerator instance
    """
    # Configure logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Don't set CUDA_VISIBLE_DEVICES - let Accelerate handle device management
    if "CUDA_VISIBLE_DEVICES" in os.environ and args.gpu_ids is not None:
        logger.warning("Both CUDA_VISIBLE_DEVICES and --gpu_ids are set. Using CUDA_VISIBLE_DEVICES.")

    # Create checkpoint configuration
    project_config = None
    if args.output_dir:
        project_config = ProjectConfiguration(
            project_dir=args.output_dir,
            logging_dir=os.path.join(args.output_dir, "logs")
        )

    # Configure accelerator with the right settings
    accelerator_config = {
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "mixed_precision": 'fp16' if args.fp16 else 'no',
        "project_config": project_config,
    }

    if args.log_with_tensorboard:
        accelerator_config["log_with"] = "tensorboard"

    # Create accelerator
    accelerator = Accelerator(**accelerator_config)

    # Set proper logging level based on process role
    logger.setLevel(logging.INFO if accelerator.is_main_process else logging.WARNING)

    # Log accelerator configuration on main process
    if accelerator.is_main_process:
        logger.info("-" * 50)
        logger.info("Accelerator Configuration:")
        logger.info(f"  Process index: {accelerator.process_index}")
        logger.info(f"  Local process index: {accelerator.local_process_index}")
        logger.info(f"  Number of processes: {accelerator.num_processes}")
        logger.info(f"  Distributed type: {accelerator.distributed_type}")
        logger.info(f"  Mixed precision: {accelerator.mixed_precision}")
        logger.info(f"  Device: {accelerator.device}")

        # Log GPU info if available
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                if i == accelerator.local_process_index or accelerator.num_processes == 1:
                    device_name = torch.cuda.get_device_name(i)
                    total_memory = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
                    logger.info(f"  GPU {i}: {device_name} ({total_memory:.2f} GB)")

        logger.info("-" * 50)

    return accelerator


def save_checkpoint(accelerator, args, epoch, step, model, optimizer, lr_scheduler, tokenizer):
    """
    Save a checkpoint with proper state management.

    Args:
        accelerator: Accelerator instance
        args: Command-line arguments
        epoch: Current epoch
        step: Current step
        model: Model to save
        optimizer: Optimizer to save
        lr_scheduler: Learning rate scheduler to save
        tokenizer: Tokenizer to save
    """
    # Determine checkpoint path (epoch or step-based)
    if step % args.save_steps == 0:
        checkpoint_path = os.path.join(args.output_dir, f"step_{step}")
    else:
        checkpoint_path = os.path.join(args.output_dir, f"epoch_{epoch}")

    logger.info(f"Saving checkpoint to {checkpoint_path}")

    # Use Accelerate's state saving
    accelerator.save_state(checkpoint_path)

    # Save tokenizer on main process only
    if accelerator.is_main_process and tokenizer is not None:
        tokenizer.save_pretrained(os.path.join(checkpoint_path, "tokenizer"))

    # Clean up old checkpoints (only keep N most recent)
    if accelerator.is_main_process and args.save_total_limit is not None:
        cleanup_checkpoints(args.output_dir, args.save_total_limit)

    # Wait for checkpoint saving to complete
    accelerator.wait_for_everyone()


def cleanup_checkpoints(output_dir, save_total_limit):
    """
    Clean up old checkpoints, keeping only the most recent ones.

    Args:
        output_dir: Directory containing checkpoints
        save_total_limit: Maximum number of checkpoints to keep
    """
    import os
    import shutil
    import re

    # Get all checkpoint directories
    checkpoint_dirs = []

    # Match both epoch and step directories
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
    """
    Resume training from a checkpoint with proper state tracking.

    Args:
        accelerator: Accelerator instance
        args: Command-line arguments
        num_update_steps_per_epoch: Number of update steps per epoch

    Returns:
        Tuple[int, int]: Starting epoch and completed steps
    """
    import os
    from pathlib import Path

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

    # Check for optimizer and scheduler states
    optimizer_path = os.path.join(checkpoint_path, "optimizer.pt")
    scheduler_path = os.path.join(checkpoint_path, "scheduler.pt")

    # Load checkpoint state
    accelerator.load_state(checkpoint_path)

    # Verify that optimizer and scheduler states were loaded
    if accelerator.is_main_process:
        if os.path.exists(optimizer_path):
            logger.info("Optimizer state loaded successfully")
        else:
            logger.warning("Optimizer state not found in checkpoint")

        if os.path.exists(scheduler_path):
            logger.info("Scheduler state loaded successfully")
        else:
            logger.warning("Scheduler state not found in checkpoint")

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
    """
    Save the model and tokenizer using Accelerate.

    Args:
        accelerator: Accelerator instance
        model: Model to save
        tokenizer: Tokenizer to save
        output_dir: Directory to save to
    """
    # Create output directory if needed
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)

    # Wait for directory creation
    accelerator.wait_for_everyone()

    # Unwrap model and save on main process
    unwrapped_model = accelerator.unwrap_model(model)

    if accelerator.is_main_process:
        # Save model and config using HuggingFace's built-in methods
        unwrapped_model.save_pretrained(output_dir)
        logger.info(f"Model saved to {output_dir}")

        # Save tokenizer if provided
        if tokenizer is not None:
            tokenizer_path = os.path.join(output_dir, "tokenizer")
            tokenizer.save_pretrained(tokenizer_path)
            logger.info(f"Tokenizer saved to {tokenizer_path}")

    # Wait for saving to complete
    accelerator.wait_for_everyone()


def verify_model_tokenizer_compatibility(model, tokenizer):
    """
    Verify that model and tokenizer have compatible vocabulary sizes.
    Resizes model embeddings if necessary.

    Args:
        model: The model to verify
        tokenizer: The tokenizer to verify

    Returns:
        The model with potentially resized embeddings
    """
    tokenizer_vocab_size = tokenizer.vocab_size
    model_vocab_size = model.config.vocab_size
    embedding_size = model.get_input_embeddings().weight.shape[0]

    logger.info(f"COMPATIBILITY CHECK:")
    logger.info(f"  Tokenizer vocab size: {tokenizer_vocab_size}")
    logger.info(f"  Model config vocab size: {model_vocab_size}")
    logger.info(f"  Model embedding size: {embedding_size}")

    # Check tokenizer and model config
    if tokenizer_vocab_size != model_vocab_size:
        logger.warning(f"Tokenizer vocab size ({tokenizer_vocab_size}) doesn't match model config ({model_vocab_size})")

        # Attempt to fix by resizing embeddings
        logger.info(f"Resizing model embeddings to match tokenizer vocab size: {tokenizer_vocab_size}")
        model.resize_token_embeddings(tokenizer_vocab_size)

        # Check if fix worked
        new_embedding_size = model.get_input_embeddings().weight.shape[0]
        if new_embedding_size != tokenizer_vocab_size:
            raise ValueError(
                f"Failed to resize embeddings. Tokenizer: {tokenizer_vocab_size}, "
                f"Model: {new_embedding_size}"
            )
        else:
            logger.info(f"Successfully resized embeddings to {new_embedding_size}")

    # Final check of embedding size
    if embedding_size != tokenizer_vocab_size:
        logger.warning(f"Model embedding size ({embedding_size}) doesn't match tokenizer ({tokenizer_vocab_size})")
        logger.info(f"Resizing token embeddings to match tokenizer: {tokenizer_vocab_size}")
        model.resize_token_embeddings(tokenizer_vocab_size)

    return model


def safe_training_step(model, batch, accelerator, args):
    """
    Execute a single training step with robust error handling.

    Args:
        model: The model to train
        batch: The batch of data
        accelerator: Accelerator instance
        args: Command-line arguments

    Returns:
        torch.Tensor or None: The loss value if successful, None otherwise
    """
    try:
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
                else:
                    logger.info(f"Processing long sequence batch (length: {seq_length})")

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
    """
    Process a batch in smaller micro-batches to handle memory constraints.

    Args:
        model: The model to train
        batch: The batch of data
        accelerator: Accelerator instance
        args: Command-line arguments
        micro_batch_size: Size of each micro-batch

    Returns:
        torch.Tensor: The accumulated loss
    """
    # Get the original batch size
    full_batch_size = batch['input_ids'].shape[0]

    # If batch size is already 1, we can't split further
    if full_batch_size <= micro_batch_size:
        raise ValueError("Batch size is already at minimum, cannot process in micro-batches")

    # Initialize accumulated loss
    accumulated_loss = torch.tensor(0.0, device=accelerator.device)

    # Process each micro-batch
    for i in range(0, full_batch_size, micro_batch_size):
        # Get micro-batch indices
        end_idx = min(i + micro_batch_size, full_batch_size)
        logger.info(
            f"Processing micro-batch {i // micro_batch_size + 1}/{(full_batch_size + micro_batch_size - 1) // micro_batch_size}")

        # Create micro-batch
        micro_batch = {k: v[i:end_idx] for k, v in batch.items()}

        # Forward pass
        with torch.cuda.amp.autocast(enabled=args.fp16):
            outputs = model(**micro_batch)
            micro_loss = outputs.loss / args.gradient_accumulation_steps / (full_batch_size / micro_batch_size)

        # Backward pass
        accelerator.backward(micro_loss)

        # Add to accumulated loss
        accumulated_loss += micro_loss.detach() * (end_idx - i)

    # Return average loss
    return accumulated_loss / full_batch_size


def calculate_training_steps(train_dataloader, gradient_accumulation_steps, num_epochs):
    """
    Calculate the number of update steps for training.

    Args:
        train_dataloader: Training data loader
        gradient_accumulation_steps: Number of steps for gradient accumulation
        num_epochs: Number of training epochs

    Returns:
        Tuple[int, int]: Steps per epoch and total training steps
    """
    # Calculate steps per epoch with proper ceiling division
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)

    # Calculate total training steps
    max_train_steps = num_epochs * num_update_steps_per_epoch

    return num_update_steps_per_epoch, max_train_steps


def train_with_accelerate(args):
    """
    Main training function using Accelerate.

    Args:
        args: Command-line arguments
    """
    # Use the setup function to initialize accelerator
    accelerator = setup_accelerator(args)

    # Get the correct device from accelerator
    device = accelerator.device
    logger = logging.getLogger(__name__)

    # Set random seed for reproducibility
    if args.seed is not None:
        from accelerate.utils import set_seed
        set_seed(args.seed)

    # Create output directory
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # Wait for all processes to sync
    accelerator.wait_for_everyone()

    # Import here to avoid early initialization
    from transformers import BertForMaskedLM
    from torch.utils.data import DataLoader


    # Setup tokenizer
    tokenizer = setup_tokenizer(args, accelerator)

    # Setup model
    if args.model_path and os.path.exists(args.model_path):
        logger.info(f"Loading existing model from {args.model_path}")
        model = BertForMaskedLM.from_pretrained(args.model_path)
        if not hasattr(model.config, 'use_alibi') or not model.config.use_alibi:
            logger.info("Adding ALiBi to loaded model")
            model = modify_bert_for_alibi(model)
    else:
        logger.info("Initializing new BERT model with ALiBi for genomic data")
        config = create_genomic_bert_config(
            vocab_size=tokenizer.vocab_size,
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            num_attention_heads=args.num_attention_heads,
            intermediate_size=args.hidden_size * 4,
            max_position_embeddings=args.pre_training_length,
            use_alibi=True,
            max_supported_length=args.max_supported_model_length,
        )
        model = create_genomic_bert_model(config)

    # Check model and tokenizer compatibility
    model = verify_model_tokenizer_compatibility(model, tokenizer)

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
    )

    # Data collator for masked language modeling
    data_collator = GenomicMLMDataCollator(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=args.mlm_probability
    )

    # Configure data loader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=data_collator,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        **({"prefetch_factor": 2} if args.num_workers > 0 else {}),
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

    # Prepare with accelerator
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )


    test_lengths = [
        args.pre_training_length,
        args.pre_training_length * 2,
        args.pre_training_length * 4,
    ]
    if args.max_inference_length:
        test_lengths.append(args.max_inference_length)

    extrapolation_test_seqs = generate_test_sequences(test_lengths)

    # Test initial extrapolation
    if accelerator.is_main_process:
        logger.info("\nInitial length extrapolation test:")
        test_sequence_length_extrapolation(accelerator, model, tokenizer, extrapolation_test_seqs)

    # Resume from checkpoint if needed
    starting_epoch, completed_steps = resume_from_checkpoint(accelerator, args, num_update_steps_per_epoch)

    # Compute total training batch size
    total_batch_size = args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    # Log training information
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num epochs = {args.epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    # Setup progress bar (only on main process)
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_main_process)
    progress_bar.update(completed_steps)

    # Main training loop
    for epoch in range(starting_epoch, args.epochs):
        model.train()
        total_loss = 0
        valid_steps = 0  # Count successful steps for averaging

        # Report GPU memory status on main process
        if torch.cuda.is_available() and accelerator.is_main_process:
            logger.info(f"Starting epoch {epoch} - GPU memory status:")
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / (1024 * 1024)
                reserved = torch.cuda.memory_reserved(i) / (1024 * 1024)
                logger.info(f"  GPU {i}: Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")

        # Process batches with improved error handling
        for step, batch in enumerate(train_dataloader):
            # Special handling for first few batches in first epoch
            if epoch == 0 and step < 5:
                logger.info(f"Carefully processing batch {step} in first epoch")
                # Extra safety for first batches - limit sequence length
                if 'input_ids' in batch:
                    seq_length = batch['input_ids'].size(1)
                    logger.info(f"  Sequence length: {seq_length}")
                    if seq_length > 512:  # Be very conservative initially
                        logger.warning(f"  First batch has long sequences ({seq_length}). Truncating to 512.")
                        batch = {k: v[:, :512] if v.dim() > 1 else v for k, v in batch.items()}

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

    # Save final model
    if accelerator.is_main_process:
        final_dir = os.path.join(args.output_dir, "final")
        save_model(accelerator, model, tokenizer, final_dir)

    # Final length extrapolation test (only on main process)
    if accelerator.is_main_process:
        logger.info("\nFinal length extrapolation test:")
        test_sequence_length_extrapolation(accelerator, model, tokenizer, extrapolation_test_seqs)

    logger.info("Training complete!")