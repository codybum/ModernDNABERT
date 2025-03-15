#!/usr/bin/env python
"""
Standalone script for training a genomic tokenizer with optimized CPU performance.

This script is completely independent from the BERT training process and focuses
on efficiently training a SentencePiece BPE tokenizer on genomic sequences using
all available CPU cores.
"""

import os
import argparse
import logging
import time
import multiprocessing
import random
import psutil
import tqdm
import sys
from typing import List, Optional, Dict, Any, Union, Tuple
from dataclasses import dataclass

# Import only what we need directly
from tokenization.genomic_tokenizer import GenomicTokenizer, _ensure_special_tokens

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


@dataclass
class ProcessingStats:
    """Class to track processing statistics"""
    start_time: float = 0.0
    sequences_processed: int = 0
    bytes_processed: int = 0
    files_processed: int = 0
    chunks_processed: int = 0
    invalid_sequences: int = 0


# Progress tracking for multiprocessing
stats = ProcessingStats()
progress_bar = None


def process_fasta_chunk(chunk_data):
    """Process a chunk of a FASTA file to extract valid genomic sequences."""
    file_path, chunk_id, start_pos, end_pos, file_size = chunk_data
    sequences = []
    current_seq = ""

    try:
        with open(file_path, 'r') as f:
            f.seek(start_pos)

            # If not at start of file, read until next newline to avoid partial line
            if start_pos > 0:
                f.readline()

            # Read chunk
            bytes_read = 0
            while f.tell() < end_pos and f.tell() < file_size:
                line = f.readline()
                bytes_read += len(line)

                if not line:
                    break

                line = line.strip()
                # Skip FASTA headers
                if line.startswith('>'):
                    if current_seq:
                        sequences.append(current_seq)
                        current_seq = ""
                    continue
                # Only accept A, T, G, C sequences, ignore others
                if set(line.upper()) <= set('ATGC'):
                    current_seq += line.upper()

            # Add the last sequence if exists
            if current_seq:
                sequences.append(current_seq)

        return {
            'sequences': sequences,
            'chunk_id': chunk_id,
            'bytes_processed': bytes_read,
            'num_sequences': len(sequences)
        }
    except Exception as e:
        logger.error(f"Error processing chunk {chunk_id} of file {file_path}: {e}")
        return {
            'sequences': [],
            'chunk_id': chunk_id,
            'bytes_processed': 0,
            'num_sequences': 0
        }


def update_progress(result):
    """Callback function to update progress bar"""
    global stats, progress_bar

    if progress_bar is not None:
        stats.sequences_processed += result['num_sequences']
        stats.bytes_processed += result['bytes_processed']
        stats.chunks_processed += 1

        progress_bar.update(result['bytes_processed'])
        progress_bar.set_description(
            f"Processed {stats.chunks_processed} chunks, {stats.sequences_processed} sequences"
        )


def split_file_into_chunks(file_path, chunk_size=10 * 1024 * 1024):
    """Split a file into chunks for parallel processing."""
    file_size = os.path.getsize(file_path)
    chunks = []

    for chunk_id, start_pos in enumerate(range(0, file_size, chunk_size)):
        end_pos = min(start_pos + chunk_size, file_size)
        chunks.append((file_path, chunk_id, start_pos, end_pos, file_size))

    return chunks


def cpu_optimized_prepare_data(input_files: List[str], output_file: str, sample_size: Optional[int] = None,
                               num_workers: int = None, seq_chunk_size: int = 1000, stride: int = 500,
                               max_safe_sequence_length: int = 50000):
    """Prepare genomic data for tokenizer training using parallel processing."""
    global stats, progress_bar

    # Use all available cores if not specified
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()

    # Calculate memory size and core count for optimal chunking
    mem_info = psutil.virtual_memory()
    total_mem_gb = mem_info.total / (1024 ** 3)

    # Adjust workers based on memory
    effective_workers = min(num_workers, max(1, int(total_mem_gb / 2)))
    if effective_workers < num_workers:
        logger.info(f"Limiting workers to {effective_workers} based on available memory ({total_mem_gb:.1f} GB)")

    num_workers = effective_workers

    # Initialize progress tracking
    stats = ProcessingStats()
    stats.start_time = time.time()

    # Calculate total size for progress bar
    total_size = sum(os.path.getsize(f) for f in input_files)

    logger.info(f"Preparing genomic data using {num_workers} worker processes")
    logger.info(f"Total data size: {total_size / (1024 ** 2):.2f} MB")

    # Set up progress bar
    progress_bar = tqdm.tqdm(total=total_size, unit='B', unit_scale=True, desc="Processing files")

    all_sequences = []
    file_chunk_size = 10 * 1024 * 1024  # 10MB chunks for file processing

    # Create multiprocessing pool
    with multiprocessing.Pool(processes=num_workers) as pool:
        for file_path in input_files:
            try:
                file_size = os.path.getsize(file_path)
                stats.files_processed += 1

                if file_size == 0:
                    logger.warning(f"Skipping empty file: {file_path}")
                    continue

                logger.info(
                    f"Processing file {stats.files_processed}/{len(input_files)}: {file_path} ({file_size / (1024 ** 2):.2f} MB)")

                # Divide file into chunks for parallel processing
                chunks = split_file_into_chunks(file_path, file_chunk_size)
                logger.info(f"File divided into {len(chunks)} chunks for parallel processing")

                # Process chunks in parallel with callback for progress updates
                results = []
                for chunk in chunks:
                    result = pool.apply_async(
                        process_fasta_chunk,
                        args=(chunk,),
                        callback=update_progress
                    )
                    results.append(result)

                # Collect results
                for result in results:
                    result_data = result.get()
                    all_sequences.extend(result_data['sequences'])

            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")

    # Close progress bar
    progress_bar.close()
    progress_bar = None

    # Report statistics
    processing_time = time.time() - stats.start_time
    logger.info(f"Extracted {len(all_sequences)} sequences in {processing_time:.2f} seconds")
    logger.info(f"Processing speed: {stats.bytes_processed / processing_time / (1024 ** 2):.2f} MB/s")

    # Sample if needed
    if sample_size and sample_size < len(all_sequences):
        logger.info(f"Sampling {sample_size} sequences from {len(all_sequences)} total sequences")
        all_sequences = random.sample(all_sequences, sample_size)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Apply sequence length safety limit
    long_sequences = 0
    for i, seq in enumerate(all_sequences):
        if len(seq) > max_safe_sequence_length:
            all_sequences[i] = seq[:max_safe_sequence_length]
            long_sequences += 1

    if long_sequences > 0:
        logger.warning(
            f"Truncated {long_sequences} sequences that exceeded max_safe_sequence_length ({max_safe_sequence_length})")

    # Write to output file with chunking for better SentencePiece training using stride
    with open(output_file, 'w', buffering=16 * 1024 * 1024) as f:  # 16MB buffer
        for seq in all_sequences:
            # Process with sliding window for overlapping chunks
            for i in range(0, len(seq) - seq_chunk_size + 1, stride):
                chunk = seq[i:i + seq_chunk_size]
                if len(chunk) >= 50:  # Only keep reasonably sized chunks
                    f.write(chunk + '\n')

            # Handle remaining sequence at the end if not fully covered
            if len(seq) % stride != 0 and len(seq) > seq_chunk_size:
                last_chunk = seq[-seq_chunk_size:]
                if len(last_chunk) >= 50:
                    f.write(last_chunk + '\n')

    logger.info(f"Prepared data saved to {output_file}")
    return output_file


def cpu_optimized_train_sentencepiece(input_file: str, model_prefix: str, vocab_size: int = 4096,
                                      num_threads: int = None, max_sentence_length: int = 2048,
                                      character_coverage: float = 1.0, model_type: str = "bpe"):
    """Train a SentencePiece BPE tokenizer using multiple CPU threads."""
    import sentencepiece as spm

    # Verify input file
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    file_size = os.path.getsize(input_file)
    if file_size == 0:
        raise ValueError(f"Input file is empty: {input_file}")

    # Use all CPU cores if not specified
    if num_threads is None:
        num_threads = multiprocessing.cpu_count()

    logger.info(f"Training SentencePiece with {num_threads} threads")

    # SentencePiece training parameters optimized for genomic data and CPU efficiency
    training_args = {
        'input': input_file,
        'model_prefix': model_prefix,
        'vocab_size': vocab_size,
        'character_coverage': character_coverage,
        'model_type': model_type,
        'input_sentence_size': 10000000,  # Process many sequences
        'shuffle_input_sentence': True,
        'normalization_rule_name': 'identity',  # No normalization for DNA
        'byte_fallback': False,  # DNA has only 4 letters
        'unk_id': 0,
        'bos_id': 1,
        'eos_id': 2,
        'pad_id': 3,
        'num_threads': num_threads,  # Multi-threading
        'max_sentence_length': max_sentence_length,  # Limit sequence processing length
        'train_extremely_large_corpus': file_size > 1_000_000_000,  # Enable for very large files
        'input_format': 'text',  # Format is plain text with one sentence per line
        'split_by_whitespace': False  # Don't split by whitespace (DNA doesn't have spaces)
    }

    start_time = time.time()
    try:
        # Train the tokenizer with progress reporting
        logger.info("Starting SentencePiece training...")
        spm.SentencePieceTrainer.train(**training_args)
        model_file = f"{model_prefix}.model"
        vocab_file = f"{model_prefix}.vocab"

        # Verify output files were created
        if not os.path.exists(model_file) or not os.path.exists(vocab_file):
            raise FileNotFoundError(f"SentencePiece failed to create output files")

        logger.info(f"Trained SentencePiece model in {time.time() - start_time:.2f} seconds")
        return model_file, vocab_file

    except Exception as e:
        logger.error(f"SentencePiece training failed: {e}")
        raise RuntimeError(f"Failed to train SentencePiece tokenizer: {e}")


def verify_tokenizer(tokenizer, test_sequences=None):
    """Verify that the tokenizer works correctly."""
    logger.info("Verifying tokenizer functionality...")

    if test_sequences is None:
        test_sequences = [
            "ATGCATGCATGC",
            "GGGAAATTTCCC",
            "ATATATATATAT",
            "GCGCGCGCGCGC"
        ]

    # Test basic tokenization
    for seq in test_sequences:
        tokens = tokenizer.tokenize(seq)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        decoded = tokenizer.decode(token_ids)

        logger.info(f"Test sequence: {seq}")
        logger.info(f"Tokens: {tokens}")
        logger.info(f"Token IDs: {token_ids}")
        logger.info(f"Decoded: {decoded}")
        logger.info("-" * 40)

    # Test special tokens
    logger.info("Special tokens:")
    for name, token in tokenizer.special_tokens_map.items():
        token_id = tokenizer.convert_tokens_to_ids(token)
        logger.info(f"  {name}: '{token}' -> ID: {token_id}")

    return True


def main():
    """Main function for tokenizer training."""
    parser = argparse.ArgumentParser(description="Train a SentencePiece BPE tokenizer for genomic sequences")

    # Input/Output - same as train.py
    io_group = parser.add_argument_group("Input/Output")
    io_group.add_argument("--input_files", nargs="+", required=True,
                          help="Input genomic sequence files (FASTA format)")
    io_group.add_argument("--output_dir", required=True,
                          help="Output directory for tokenizer")

    # Tokenizer options - aligned with train.py
    tokenizer_group = parser.add_argument_group("Tokenizer Options")
    tokenizer_group.add_argument("--vocab_size", type=int, default=4096,
                                 help="Vocabulary size for BPE tokenizer (default: 4096)")
    tokenizer_group.add_argument("--tokenizer_sample_size", type=int, default=100000,
                                 help="Number of sequences to sample for tokenizer training (default: 100000)")
    tokenizer_group.add_argument("--character_coverage", type=float, default=1.0,
                                 help="Character coverage for SentencePiece training (default: 1.0)")
    tokenizer_group.add_argument("--sp_model_type", type=str, default="bpe", choices=["bpe", "unigram", "char", "word"],
                                 help="SentencePiece model type (default: bpe)")

    # Sequence options - aligned with train.py
    seq_group = parser.add_argument_group("Sequence Options")
    seq_group.add_argument("--chunk_size", type=int, default=2000,
                           help="Base size of sequence chunks (default: 2000)")
    seq_group.add_argument("--stride", type=int, default=1000,
                           help="Stride for overlapping chunks (default: 1000)")
    seq_group.add_argument("--max_safe_sequence_length", type=int, default=50000,
                           help="Maximum safe sequence length for processing (default: 50000)")
    seq_group.add_argument("--max_sentence_length", type=int, default=2048,
                           help="Maximum sentence length for SentencePiece training (default: 2048)")

    # Performance options
    perf_group = parser.add_argument_group("Performance Options")
    perf_group.add_argument("--num_workers", type=int, default=None,
                            help="Number of worker processes for data preparation (default: all available cores)")
    perf_group.add_argument("--num_threads", type=int, default=None,
                            help="Number of threads for SentencePiece training (default: all available cores)")

    # Other options
    other_group = parser.add_argument_group("Other Options")
    other_group.add_argument("--seed", type=int, default=42,
                             help="Random seed (default: 42)")
    other_group.add_argument("--debug", action="store_true",
                             help="Enable debug output")
    other_group.add_argument("--skip_verification", action="store_true",
                             help="Skip tokenizer verification")
    other_group.add_argument("--force", action="store_true",
                             help="Overwrite existing tokenizer files")

    args = parser.parse_args()

    # Configure logging
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")

    # Set random seed
    random.seed(args.seed)
    logger.info(f"Random seed set to {args.seed}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Check if files already exist
    tokenizer_dir = os.path.join(args.output_dir, "tokenizer")
    if os.path.exists(tokenizer_dir) and not args.force:
        logger.warning(f"Tokenizer directory already exists: {tokenizer_dir}")
        override = input("Override existing tokenizer? (y/n): ").lower().strip() == 'y'
        if not override:
            logger.info("Exiting without overwriting existing tokenizer")
            return

    # Log system info
    logger.info(f"CPU cores available: {multiprocessing.cpu_count()}")
    logger.info(f"Memory available: {psutil.virtual_memory().total / (1024 ** 3):.2f} GB")
    logger.info(f"Starting tokenizer training process...")

    total_start_time = time.time()

    try:
        # Step 1: Prepare data
        logger.info("Preparing genomic data...")
        prepared_data = cpu_optimized_prepare_data(
            args.input_files,
            os.path.join(args.output_dir, "prepared_sequences.txt"),
            args.tokenizer_sample_size,
            args.num_workers,
            args.chunk_size,
            args.stride,
            args.max_safe_sequence_length
        )

        data_prep_time = time.time()
        logger.info(f"Data preparation completed in {data_prep_time - total_start_time:.2f} seconds")

        # Step 2: Train SentencePiece model
        logger.info(f"Training tokenizer with vocab size {args.vocab_size}...")
        model_prefix = os.path.join(args.output_dir, "genomic_bpe")
        model_file, vocab_file = cpu_optimized_train_sentencepiece(
            prepared_data,
            model_prefix,
            args.vocab_size,
            args.num_threads,
            args.max_sentence_length,
            args.character_coverage,
            args.sp_model_type
        )

        training_time = time.time()
        logger.info(f"SentencePiece training completed in {training_time - data_prep_time:.2f} seconds")

        # Step 3: Create and configure GenomicTokenizer
        logger.info("Creating GenomicTokenizer with special tokens...")
        tokenizer = GenomicTokenizer(
            sentencepiece_model_file=model_file,
            unk_token="<unk>",
            pad_token="<pad>",
            mask_token="<mask>"
        )

        # Ensure special tokens are properly configured
        _ensure_special_tokens(tokenizer)

        # Step 4: Save tokenizer
        tokenizer_output_dir = os.path.join(args.output_dir, "tokenizer")
        os.makedirs(tokenizer_output_dir, exist_ok=True)
        tokenizer.save_pretrained(tokenizer_output_dir)
        logger.info(f"Tokenizer saved to {tokenizer_output_dir}")

        # Step 5: Verify tokenizer (optional)
        if not args.skip_verification:
            verify_tokenizer(tokenizer)

        # Log performance summary
        end_time = time.time()
        logger.info(f"Tokenizer training completed in {end_time - total_start_time:.2f} seconds")
        logger.info(f"  - Data preparation: {data_prep_time - total_start_time:.2f} seconds")
        logger.info(f"  - SentencePiece training: {training_time - data_prep_time:.2f} seconds")
        logger.info(f"  - Tokenizer finalization: {end_time - training_time:.2f} seconds")

        logger.info(f"Tokenizer is ready for use with BERT training!")

    except Exception as e:
        logger.error(f"Error during tokenizer training: {e}")
        logger.exception("Detailed exception information:")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())