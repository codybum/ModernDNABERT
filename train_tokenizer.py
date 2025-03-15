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
from typing import List, Optional

# Import only what we need directly
from tokenization.genomic_tokenizer import GenomicTokenizer, _ensure_special_tokens

logger = logging.getLogger(__name__)


# CPU-optimized functions that don't depend on or interfere with Accelerate
def process_fasta_file(file_path):
    """Process a single FASTA file and extract valid genomic sequences."""
    sequences = []
    try:
        with open(file_path, 'r') as f:
            current_seq = ""
            for line in f:
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

        return sequences
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}")
        return []


def cpu_optimized_prepare_data(input_files: List[str], output_file: str, sample_size: Optional[int] = None,
                               num_workers: int = None):
    """Prepare genomic data for tokenizer training using parallel processing."""
    # Use all available cores if not specified
    if num_workers is None:
        num_workers = min(multiprocessing.cpu_count(), len(input_files))

    logger.info(f"Preparing genomic data using {num_workers} worker processes")
    start_time = time.time()

    # Process files in parallel
    all_sequences = []
    if num_workers > 1 and len(input_files) > 1:
        with multiprocessing.Pool(processes=num_workers) as pool:
            results = pool.map(process_fasta_file, input_files)

        # Flatten the results
        for seqs in results:
            all_sequences.extend(seqs)
    else:
        # Single-threaded fallback
        for file_path in input_files:
            all_sequences.extend(process_fasta_file(file_path))

    logger.info(f"Extracted {len(all_sequences)} sequences in {time.time() - start_time:.2f} seconds")

    # Sample if needed
    if sample_size and sample_size < len(all_sequences):
        logger.info(f"Sampling {sample_size} sequences from {len(all_sequences)} total sequences")
        all_sequences = random.sample(all_sequences, sample_size)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Write to output file with chunking for better SentencePiece training
    chunk_size = 1000
    with open(output_file, 'w', buffering=8 * 1024 * 1024) as f:  # 8MB buffer
        for seq in all_sequences:
            # Split long sequences into manageable chunks
            for i in range(0, len(seq), chunk_size):
                chunk = seq[i:i + chunk_size]
                if len(chunk) > 50:  # Only keep reasonably sized chunks
                    f.write(chunk + '\n')

    logger.info(f"Prepared data saved to {output_file}")
    return output_file


def cpu_optimized_train_sentencepiece(input_file: str, model_prefix: str, vocab_size: int = 4096,
                                      num_threads: int = None, max_sentence_length: int = 2048):
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
        'character_coverage': 1.0,  # Full coverage for DNA bases
        'model_type': 'bpe',
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
        'train_extremely_large_corpus': file_size > 1_000_000_000  # Enable for very large files
    }

    start_time = time.time()
    try:
        # Train the tokenizer
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


def main():
    """Main function for tokenizer training."""
    parser = argparse.ArgumentParser(description="Train a SentencePiece BPE tokenizer for genomic sequences")

    # Input/Output
    parser.add_argument("--input_files", nargs="+", required=True,
                        help="Input genomic sequence files (FASTA format)")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory for tokenizer")

    # Tokenizer options
    parser.add_argument("--vocab_size", type=int, default=4096,
                        help="Vocabulary size for BPE tokenizer (default: 4096)")
    parser.add_argument("--tokenizer_sample_size", type=int, default=100000,
                        help="Number of sequences to sample for tokenizer training")
    parser.add_argument("--max_sentence_length", type=int, default=2048,
                        help="Maximum sentence length for SentencePiece training (default: 2048)")

    # Performance options
    parser.add_argument("--num_data_workers", type=int, default=None,
                        help="Number of worker processes for data preparation (default: all available cores)")
    parser.add_argument("--num_training_threads", type=int, default=None,
                        help="Number of threads for SentencePiece training (default: all available cores)")

    # Other options
    parser.add_argument("--debug", action="store_true", help="Enable debug output")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.DEBUG if args.debug else logging.INFO
    )

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Log system info
    logger.info(f"CPU cores available: {multiprocessing.cpu_count()}")
    logger.info(f"Starting tokenizer training process...")

    total_start_time = time.time()

    # Step 1: Prepare data
    logger.info("Preparing genomic data...")
    prepared_data = cpu_optimized_prepare_data(
        args.input_files,
        os.path.join(args.output_dir, "prepared_sequences.txt"),
        args.tokenizer_sample_size,
        args.num_data_workers
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
        args.num_training_threads,
        args.max_sentence_length
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

    # Test tokenization
    test_sequence = "ATGCATGCATGC"
    tokens = tokenizer.tokenize(test_sequence)
    logger.info(f"Test tokenization of '{test_sequence}': {tokens}")

    # Log performance summary
    end_time = time.time()
    logger.info(f"Tokenizer training completed in {end_time - total_start_time:.2f} seconds")
    logger.info(f"  - Data preparation: {data_prep_time - total_start_time:.2f} seconds")
    logger.info(f"  - SentencePiece training: {training_time - data_prep_time:.2f} seconds")
    logger.info(f"  - Tokenizer finalization: {end_time - training_time:.2f} seconds")

    logger.info(f"Tokenizer is ready for use with BERT training!")


if __name__ == "__main__":
    main()