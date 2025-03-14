#!/usr/bin/env python
"""
Script for training a genomic tokenizer.

This script trains a SentencePiece BPE tokenizer on genomic sequences
and saves it to the specified output directory.
"""

import os
import argparse
import logging
from tokenization.genomic_tokenizer import prepare_genomic_data, train_sentencepiece_bpe, GenomicTokenizer

logger = logging.getLogger(__name__)


def main():
    """
    Parse arguments and train tokenizer.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Train a SentencePiece BPE tokenizer for genomic sequences"
    )

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

    # Other options
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug output")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if not args.debug else logging.DEBUG
    )

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Prepare data for tokenizer training
    logger.info("Preparing data for tokenizer training...")
    prepared_data = prepare_genomic_data(
        args.input_files,
        os.path.join(args.output_dir, "prepared_sequences.txt"),
        args.tokenizer_sample_size
    )

    # Train SentencePiece BPE tokenizer
    logger.info(f"Training tokenizer with vocab size {args.vocab_size}...")
    model_prefix = os.path.join(args.output_dir, "genomic_bpe")
    model_file, vocab_file = train_sentencepiece_bpe(
        prepared_data,
        model_prefix,
        args.vocab_size
    )

    # Initialize tokenizer with required special tokens
    logger.info("Creating GenomicTokenizer with special tokens...")
    tokenizer = GenomicTokenizer(
        sentencepiece_model_file=model_file,
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>"
    )

    # Ensure mask token exists and is properly defined
    from tokenization.genomic_tokenizer import _ensure_special_tokens
    _ensure_special_tokens(tokenizer)

    # Save the tokenizer to the specified path
    tokenizer_output_dir = os.path.join(args.output_dir, "tokenizer")
    os.makedirs(tokenizer_output_dir, exist_ok=True)
    tokenizer.save_pretrained(tokenizer_output_dir)
    logger.info(f"Trained new tokenizer and saved to {tokenizer_output_dir}")

    # Verify the tokenizer works
    test_sequence = "ATGCATGCATGC"
    tokens = tokenizer.tokenize(test_sequence)
    logger.info(f"Test tokenization of '{test_sequence}': {tokens}")

    logger.info("Tokenizer training complete!")


if __name__ == "__main__":
    main()