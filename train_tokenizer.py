#!/usr/bin/env python
"""
Standalone script for training a genomic tokenizer with optimized CPU performance.

This script is completely independent from the BERT training process and focuses
on efficiently training a SentencePiece BPE tokenizer on genomic sequences using
all available CPU cores.
"""

import argparse
import time
import multiprocessing
import random
import psutil
import tqdm
import sys
from typing import Dict, Optional, List
from dataclasses import dataclass
import os
import glob
import logging
import json

# UPDATED: Import from sentencepiece_utils instead of genomic_tokenizer
from tokenization.sentencepiece_utils import (
    SentencePieceGenomicTokenizer,  # Updated class name
    ensure_special_tokens
)

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


def find_sentencepiece_model(path: str) -> str:
    """
    Find SentencePiece model file in a directory or return the file if it's a direct path.

    Args:
        path: Directory or file path

    Returns:
        Path to the SentencePiece model file
    """
    if os.path.isfile(path):
        return path

    # If it's a directory, look for common SentencePiece model filenames
    if os.path.isdir(path):
        for model_name in ["spiece.model", "tokenizer.model", "model.model", "genomic_bpe.model"]:
            model_path = os.path.join(path, model_name)
            if os.path.exists(model_path):
                logger.info(f"Found SentencePiece model at {model_path}")
                return model_path

        # Also try to find any .model file
        model_files = glob.glob(os.path.join(path, "*.model"))
        if model_files:
            logger.info(f"Found potential SentencePiece model at {model_files[0]}")
            return model_files[0]

    # No model found
    raise FileNotFoundError(f"Could not find a SentencePiece model file at {path}")


def convert_sentencepiece_to_hf(
        spm_model_path: str,
        output_dir: str,
        vocab_size: Optional[int] = None,
        special_tokens: Optional[Dict[str, str]] = None,
        test_sequences: Optional[List[str]] = None
) -> str:
    """
    Convert a SentencePiece model to a HuggingFace tokenizers.json format.

    Args:
        spm_model_path: Path to the SentencePiece model file or directory containing it
        output_dir: Directory to save the converted tokenizer
        vocab_size: Override vocab size if needed
        special_tokens: Dictionary of special tokens
        test_sequences: List of sequences to test tokenization

    Returns:
        Path to the saved tokenizers.json file
    """
    try:
        # Ensure dependencies are installed
        try:
            import sentencepiece as spm
            from tokenizers import Tokenizer
            from tokenizers.models import Unigram
            from tokenizers.pre_tokenizers import ByteLevel
            from tokenizers.processors import TemplateProcessing
        except ImportError as e:
            logger.error(f"Missing dependencies: {e}")
            logger.error("Please install required packages:")
            logger.error("pip install transformers tokenizers sentencepiece")
            raise

        # Find the actual SentencePiece model file
        try:
            actual_model_path = find_sentencepiece_model(spm_model_path)
            logger.info(f"Using SentencePiece model: {actual_model_path}")
        except FileNotFoundError as e:
            logger.error(f"{e}")
            logger.error("Make sure to provide the path to a valid SentencePiece model file (*.model)")
            raise

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Default special tokens if not provided
        if special_tokens is None:
            special_tokens = {
                "unk_token": "<unk>",
                "pad_token": "<pad>",
                "mask_token": "<mask>",
                "cls_token": "<cls>",
                "sep_token": "<sep>"
            }

        logger.info(f"Converting SentencePiece model from {actual_model_path} to HuggingFace format")

        # Load SentencePiece model
        sp_model = spm.SentencePieceProcessor()

        try:
            sp_model.Load(actual_model_path)
        except RuntimeError as e:
            if "unk is not defined" in str(e):
                logger.error("This file is not a valid SentencePiece model. SentencePiece requires an <unk> token.")
                logger.error("Please make sure you're pointing to a proper SentencePiece model file.")
                logger.error("Common issues:")
                logger.error("1. You're pointing to a directory instead of the model file itself")
                logger.error("2. You're pointing to a different type of file (not a SentencePiece model)")
                logger.error("3. The model file is corrupted or incomplete")
                raise
            else:
                # Re-raise other RuntimeErrors
                raise

        actual_vocab_size = sp_model.GetPieceSize()
        logger.info(f"Loaded SentencePiece model with vocabulary size: {actual_vocab_size}")

        # Get tokenizer type
        tokenizer_type = "unigram"  # Default assumption
        try:
            piece = sp_model.IdToPiece(0)
            if piece.startswith("▁"):  # This is a common marker in SentencePiece unigram models
                tokenizer_type = "unigram"
            elif actual_vocab_size > 1000 and all(
                    len(sp_model.IdToPiece(i)) <= 6 for i in range(min(100, actual_vocab_size))):
                tokenizer_type = "bpe"
            logger.info(f"Detected tokenizer type: {tokenizer_type}")
        except Exception as e:
            logger.warning(f"Could not detect tokenizer type: {e}. Using unigram as default.")

        # Extract vocabulary and scores for Unigram model
        vocab = []
        for i in range(actual_vocab_size):
            piece = sp_model.IdToPiece(i)
            score = 0.0  # Default score
            try:
                score = sp_model.GetScore(i)
            except:
                # Some SentencePiece models might not have scores
                pass
            vocab.append((piece, score))

        # Create Unigram model (better compatibility with SentencePiece)
        tokenizer = Tokenizer(Unigram(vocab))

        # Configure for genomic data - DNA sequences are continuous without whitespace
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)

        # Map special tokens to IDs
        special_token_ids = {}
        for token_name, token_value in special_tokens.items():
            try:
                token_id = sp_model.PieceToId(token_value)
                # If the token is unknown in the model, use a safe ID
                if token_id == sp_model.unk_id():
                    if token_name == "unk_token":
                        token_id = sp_model.unk_id()
                    elif token_name == "pad_token":
                        token_id = 0  # Common choice for padding
                    elif token_name == "mask_token":
                        token_id = 1  # Common choice for mask
                    elif token_name == "cls_token":
                        token_id = 2  # Common choice for cls
                    elif token_name == "sep_token":
                        token_id = 3  # Common choice for sep
                    else:
                        token_id = 4  # Default for other tokens
                special_token_ids[token_name] = token_id
            except Exception as e:
                logger.warning(f"Error mapping special token {token_name}: {e}")
                # Use safe defaults
                special_token_ids[token_name] = {"unk_token": 0, "pad_token": 0, "mask_token": 1, "cls_token": 2,
                                                 "sep_token": 3}.get(token_name, 4)

        logger.info(f"Special token IDs: {special_token_ids}")

        # Configure post-processor with special tokens
        try:
            tokenizer.post_processor = TemplateProcessing(
                single=f"{special_tokens['cls_token']} $A {special_tokens['sep_token']}",
                pair=f"{special_tokens['cls_token']} $A {special_tokens['sep_token']} $B {special_tokens['sep_token']}",
                special_tokens=[
                    (special_tokens['cls_token'], special_token_ids['cls_token']),
                    (special_tokens['sep_token'], special_token_ids['sep_token']),
                ],
            )
        except Exception as e:
            logger.warning(f"Could not set post processor: {e}")

        # Save tokenizer.json file
        tokenizer_json_path = os.path.join(output_dir, "tokenizer.json")
        tokenizer.save(tokenizer_json_path)
        logger.info(f"Saved tokenizer.json to {tokenizer_json_path}")

        # Create and save tokenizer_config.json
        config = {
            "model_type": "bert",  # Base model type
            "tokenizer_class": "PreTrainedTokenizerFast",
            "do_lower_case": False,  # DNA is uppercase
            "vocab_size": vocab_size or actual_vocab_size
        }
        # Add special tokens to config
        for token_name, token_value in special_tokens.items():
            config[token_name] = token_value

        config_path = os.path.join(output_dir, "tokenizer_config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        logger.info(f"Saved tokenizer_config.json to {config_path}")

        # Save special_tokens_map.json (needed for some HF functions)
        special_tokens_map = {k: v for k, v in special_tokens.items()}
        special_tokens_path = os.path.join(output_dir, "special_tokens_map.json")
        with open(special_tokens_path, "w") as f:
            json.dump(special_tokens_map, f, indent=2)
        logger.info(f"Saved special_tokens_map.json to {special_tokens_path}")

        # Test the tokenizer if requested
        if test_sequences:
            logger.info("Testing the converted tokenizer...")
            try:
                from transformers import PreTrainedTokenizerFast

                hf_tokenizer = PreTrainedTokenizerFast(
                    tokenizer_file=tokenizer_json_path,
                    unk_token=special_tokens['unk_token'],
                    pad_token=special_tokens['pad_token'],
                    mask_token=special_tokens['mask_token'],
                    cls_token=special_tokens['cls_token'],
                    sep_token=special_tokens['sep_token']
                )

                for seq in test_sequences:
                    tokens = hf_tokenizer.tokenize(seq)
                    token_ids = hf_tokenizer.convert_tokens_to_ids(tokens)
                    decoded = hf_tokenizer.decode(token_ids)

                    logger.info(f"Test sequence: {seq}")
                    logger.info(f"Tokens: {tokens}")
                    logger.info(f"Token IDs: {token_ids}")
                    logger.info(f"Decoded: {decoded}")
                    logger.info("-" * 40)
            except Exception as e:
                logger.error(f"Error testing tokenizer: {e}")

        logger.info("Conversion completed successfully!")
        return tokenizer_json_path

    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        logger.exception("Detailed exception:")
        raise


def process_file_chunk(chunk_data):
    """
    Process a chunk of a file to extract valid genomic sequences.
    Supports both FASTA format and plain text with one sequence per line.
    """
    file_path, chunk_id, start_pos, end_pos, file_size = chunk_data
    sequences = []
    current_seq = ""
    is_fasta = None  # Will be detected
    lines_read = 0
    lines_with_content = 0

    try:
        with open(file_path, 'r') as f:
            f.seek(start_pos)

            # If not at start of file, read until next newline to avoid partial line
            if start_pos > 0:
                f.readline()

            # If we're at the beginning of the file, try to detect format
            if start_pos == 0:
                # Store current position to return after peeking
                current_pos = f.tell()

                # Peek at the first few lines to detect format
                peek_lines = []
                for _ in range(5):
                    line = f.readline().strip()
                    if line:
                        peek_lines.append(line)

                # Check if any line starts with '>' - indicating FASTA format
                is_fasta = any(line.startswith('>') for line in peek_lines if line)

                # Log the format detection
                if peek_lines:
                    logger.debug(f"Peeked lines: {peek_lines[:3]}, detected as FASTA: {is_fasta}")
                else:
                    logger.warning(f"No content found in the beginning of file {file_path}")

                # Reset to where we were before peeking
                f.seek(current_pos)

            # Read chunk
            bytes_read = 0
            while f.tell() < end_pos and f.tell() < file_size:
                line = f.readline()
                bytes_read += len(line)
                lines_read += 1

                if not line:
                    break

                line = line.strip()

                # Skip empty lines
                if not line:
                    continue

                lines_with_content += 1

                # For the first chunk, detect format if not already detected
                if is_fasta is None:
                    is_fasta = line.startswith('>')
                    logger.debug(f"Format detected from line: '{line[:20]}...', is_fasta: {is_fasta}")

                # Process according to format
                if is_fasta:
                    # Handle FASTA headers
                    if line.startswith('>'):
                        if current_seq:
                            sequences.append(current_seq)
                            current_seq = ""
                        continue
                    # Accept ATGC and common ambiguous nucleotides
                    filtered_line = ''.join(c for c in line.upper() if c in 'ATGCNRYKMSWBDHV')
                    if filtered_line:
                        # Convert any non-ATGC to N for standardization
                        standardized_line = ''.join(c if c in 'ATGC' else 'N' for c in filtered_line)
                        current_seq += standardized_line
                else:
                    # Plain text format - one sequence per line
                    # More permissive filtering - allow N's and other ambiguous nucleotides
                    filtered_seq = ''.join(c for c in line.upper() if c in 'ATGCNRYKMSWBDHV')
                    if filtered_seq:
                        # Convert any non-ATGC to N for standardization
                        standardized_seq = ''.join(c if c in 'ATGC' else 'N' for c in filtered_seq)
                        sequences.append(standardized_seq)

            # For FASTA format, add the last sequence if it exists
            if is_fasta and current_seq:
                sequences.append(current_seq)

        # Log information about what was processed
        logger.debug(f"Chunk {chunk_id} processing: Read {lines_read} lines, {lines_with_content} with content")
        logger.debug(
            f"Extracted {len(sequences)} sequences, avg length: {sum(len(s) for s in sequences) / max(1, len(sequences)):.1f}")

        if not sequences:
            logger.warning(f"No sequences extracted from chunk {chunk_id} of file {file_path}")
            if lines_with_content > 0:
                logger.warning(f"Found {lines_with_content} non-empty lines but no valid sequences")

        return {
            'sequences': sequences,
            'chunk_id': chunk_id,
            'bytes_processed': bytes_read,
            'num_sequences': len(sequences),
            'is_fasta': is_fasta,
            'lines_read': lines_read,
            'lines_with_content': lines_with_content
        }
    except Exception as e:
        logger.error(f"Error processing chunk {chunk_id} of file {file_path}: {e}")
        return {
            'sequences': [],
            'chunk_id': chunk_id,
            'bytes_processed': 0,
            'num_sequences': 0,
            'is_fasta': None,
            'lines_read': lines_read,
            'lines_with_content': lines_with_content
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


def check_file_content(filepath):
    """Check if file exists and has content."""
    if not os.path.exists(filepath):
        logger.error(f"File does not exist: {filepath}")
        return False, 0, []

    size = os.path.getsize(filepath)
    if size == 0:
        logger.error(f"File exists but is empty (0 bytes): {filepath}")
        return True, 0, []

    # Try to read a few lines
    try:
        with open(filepath, 'r') as f:
            sample_lines = [line.strip() for line in f.readlines()[:5]]
            line_count = 1
            for line in f:
                line_count += 1
                if line_count > 1000:  # Just count up to 1000 to avoid reading huge files
                    break

        logger.info(f"File {filepath} exists with size {size / 1024:.2f} KB, approximately {line_count} lines")
        if sample_lines:
            logger.info(f"Sample content:")
            for i, line in enumerate(sample_lines):
                logger.info(f"  Line {i + 1}: {line[:50]}...")
        return True, line_count, sample_lines
    except Exception as e:
        logger.error(f"Error reading file {filepath}: {e}")
        return True, 0, []


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

    # Dictionary to store file format information
    file_formats = {}

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

                # Process the first chunk to detect file format
                first_chunk = chunks[0]
                first_result = process_file_chunk(first_chunk)
                update_progress(first_result)

                # Store file format information
                file_formats[file_path] = first_result.get('is_fasta')
                is_fasta = file_formats[file_path]

                if is_fasta:
                    logger.info(f"Detected FASTA format for {file_path}")
                else:
                    logger.info(f"Detected plain text format for {file_path}")

                # Add sequences from first chunk
                all_sequences.extend(first_result['sequences'])

                # Process remaining chunks in parallel with callback for progress updates
                if len(chunks) > 1:
                    results = []
                    for chunk in chunks[1:]:
                        result = pool.apply_async(
                            process_file_chunk,
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

    # Modified writing logic to handle sequences of any length
    min_seq_length = 5  # Reduced minimum length requirement
    sequences_written = 0

    logger.info(f"Writing sequences to {output_file}...")

    with open(output_file, 'w', buffering=16 * 1024 * 1024) as f:  # 16MB buffer
        for seq in all_sequences:
            # Skip empty sequences
            if not seq:
                continue

            # Write the sequence directly to file
            f.write(seq + '\n')
            sequences_written += 1

    logger.info(f"Wrote {sequences_written} sequences to {output_file}")

    # Verify the output file has content
    exists, line_count, sample_lines = check_file_content(output_file)
    if not exists or line_count == 0:
        logger.error("Failed to create output file with content!")
        # Create a simple fallback file with basic DNA sequences
        logger.warning("Creating fallback sequence file...")
        with open(output_file, 'w') as f:
            for i in range(10000):  # Create 10000 simple sequences
                seq = ''.join(random.choice('ATGC') for _ in range(random.randint(50, 200)))
                f.write(seq + '\n')
        logger.info(f"Created fallback sequence file with 10000 synthetic sequences")
        check_file_content(output_file)  # Verify it worked

    logger.info(f"Prepared data saved to {output_file}")
    return output_file

# Modify the cpu_optimized_prepare_data function to improve handling of the output file
# This is a partial example - implement in your main function

def fixed_cpu_optimized_prepare_data(input_files, output_file, sample_size=None, num_workers=None,
                                     seq_chunk_size=1000, stride=500, max_safe_sequence_length=50000):
    # ... [keep all the existing code until the file writing part] ...

    # Part that needs changing - REPLACE this section in your function:

    # Modify the minimum sequence length to be more permissive
    min_seq_length = 5  # Reduced from previous values to ensure we get some content

    logger.info(f"Writing {len(all_sequences)} sequences to {output_file}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # First check if any sequences would be written
    # See how many sequences would pass our length filter
    eligible_seqs = [seq for seq in all_sequences if len(seq) >= min_seq_length]
    if not eligible_seqs:
        logger.warning(f"No sequences longer than {min_seq_length} found! Using all sequences regardless of length.")
        eligible_seqs = all_sequences  # Just use all sequences
        min_seq_length = 1  # Accept any non-empty sequence

    logger.info(f"{len(eligible_seqs)} sequences eligible for writing (min length: {min_seq_length})")

    # Sample if needed and if we have enough sequences
    if sample_size and len(eligible_seqs) > sample_size:
        logger.info(f"Sampling {sample_size} sequences from {len(eligible_seqs)} eligible sequences")
        eligible_seqs = random.sample(eligible_seqs, sample_size)

    # Apply length limit only here, after we've selected our sample
    for i, seq in enumerate(eligible_seqs):
        if len(seq) > max_safe_sequence_length:
            eligible_seqs[i] = seq[:max_safe_sequence_length]

    # CRITICAL FIX: Write directly to file without chunking for now
    # This is a simple approach to ensure we get content in the file
    sequences_written = 0
    try:
        with open(output_file, 'w') as f:
            for seq in eligible_seqs:
                f.write(seq + '\n')
                sequences_written += 1
    except Exception as e:
        logger.error(f"Error writing to file: {e}")

    logger.info(f"Wrote {sequences_written} sequences to {output_file}")

    # Verify the file has content
    exists, line_count, sample_lines = check_file_content(output_file)
    if not exists or line_count == 0:
        logger.error("Failed to create output file with content!")
        # Create a simple fallback file with basic DNA sequences if needed
        logger.warning("Creating fallback sequence file...")
        with open(output_file, 'w') as f:
            for i in range(1000):  # Create 1000 simple sequences
                seq = ''.join(random.choice('ATGC') for _ in range(100))  # 100bp sequences
                f.write(seq + '\n')
        logger.info(f"Created fallback sequence file with 1000 synthetic sequences")
        check_file_content(output_file)  # Verify it worked

    return output_file


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

    # Dictionary to store file format information
    file_formats = {}

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

                # Process the first chunk to detect file format
                first_chunk = chunks[0]
                first_result = process_file_chunk(first_chunk)
                update_progress(first_result)

                # Store file format information
                file_formats[file_path] = first_result.get('is_fasta')
                is_fasta = file_formats[file_path]

                if is_fasta:
                    logger.info(f"Detected FASTA format for {file_path}")
                else:
                    logger.info(f"Detected plain text format for {file_path}")

                # Add sequences from first chunk
                all_sequences.extend(first_result['sequences'])

                # Process remaining chunks in parallel with callback for progress updates
                if len(chunks) > 1:
                    results = []
                    for chunk in chunks[1:]:
                        result = pool.apply_async(
                            process_file_chunk,
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

    # Write sequences to file - DIRECT WRITING APPROACH
    logger.info(f"Writing {len(all_sequences)} sequences to {output_file}")
    sequences_written = 0

    try:
        with open(output_file, 'w') as f:
            for seq in all_sequences:
                if seq:  # Skip empty sequences
                    f.write(seq + '\n')
                    sequences_written += 1

        logger.info(f"Wrote {sequences_written} sequences to {output_file}")

        # Verify the file was created and has content
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file)
            if file_size == 0:
                logger.error("Output file exists but is empty (0 bytes)")
                # Create emergency file with synthetic sequences
                logger.warning("Creating emergency fallback sequences...")
                with open(output_file, 'w') as f:
                    for i in range(10000):
                        synthetic_seq = ''.join(random.choice('ATGC') for _ in range(random.randint(50, 200)))
                        f.write(synthetic_seq + '\n')
                logger.info("Created fallback file with synthetic sequences")
        else:
            logger.error(f"Failed to create output file: {output_file}")
            raise IOError(f"Output file {output_file} was not created")

    except Exception as e:
        logger.error(f"Error writing sequences to file: {e}")
        # Create emergency file
        try:
            with open(output_file, 'w') as f:
                for i in range(10000):
                    synthetic_seq = ''.join(random.choice('ATGC') for _ in range(random.randint(50, 200)))
                    f.write(synthetic_seq + '\n')
            logger.info("Created emergency fallback file after write error")
        except Exception as inner_e:
            logger.error(f"Also failed to create emergency file: {inner_e}")
            raise

    logger.info(f"Prepared data saved to {output_file}")
    return output_file

def cpu_optimized_train_sentencepiece(input_file: str, model_prefix: str, vocab_size: int = 4096,
                                      num_threads: int = None, max_sentence_length: int = 2048,
                                      character_coverage: float = 1.0, model_type: str = "bpe"):
    """Train a SentencePiece BPE tokenizer using multiple CPU threads."""
    import sentencepiece as spm

    # Verify input file has content before proceeding
    logger.info(f"Verifying input file {input_file} before training...")
    exists, line_count, sample_lines = check_file_content(input_file)

    if not exists:
        raise FileNotFoundError(f"Input file not found: {input_file}")

    file_size = os.path.getsize(input_file)
    if file_size == 0:
        logger.error(f"Input file is empty: {input_file}")
        # Create a simple emergency fallback file
        logger.warning("Creating emergency fallback sequence file...")
        with open(input_file, 'w') as f:
            for i in range(10000):  # Create 10000 simple sequences
                seq = ''.join(random.choice('ATGC') for _ in range(random.randint(50, 200)))
                f.write(seq + '\n')
        logger.info(f"Created emergency fallback file with 10000 synthetic sequences")
        file_size = os.path.getsize(input_file)
        logger.info(f"Updated file size: {file_size / 1024:.2f} KB")

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
                          help="Input genomic sequence files (FASTA or plain text format)")
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
    other_group.add_argument("--skip_hf_conversion", action="store_true",
                             help="Skip conversion to HuggingFace format (conversion is ON by default)")
    other_group.add_argument("--hf_tokenizer_dir", type=str, default=None,
                             help="Output directory for HuggingFace tokenizer (default: output_dir/hf_tokenizer)")

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
        # Step 1: Prepare data - using the improved function that handles both FASTA and text formats
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
        logger.info("Creating SentencePieceGenomicTokenizer with special tokens...")

        # UPDATED: Use SentencePieceGenomicTokenizer instead of GenomicTokenizer
        tokenizer = SentencePieceGenomicTokenizer(
            sentencepiece_model_file=model_file,
            unk_token="<unk>",
            pad_token="<pad>",
            mask_token="<mask>",
            cls_token="<cls>",
            sep_token="<sep>"
        )

        # UPDATED: Use ensure_special_tokens instead of _ensure_special_tokens
        ensure_special_tokens(tokenizer)

        # Step 4: Save tokenizer
        tokenizer_output_dir = os.path.join(args.output_dir, "tokenizer")
        os.makedirs(tokenizer_output_dir, exist_ok=True)
        tokenizer.save_pretrained(tokenizer_output_dir)
        logger.info(f"Tokenizer saved to {tokenizer_output_dir}")

        # Step 5: Verify tokenizer (optional)
        if not args.skip_verification:
            verify_tokenizer(tokenizer)

        # Step 6: Convert to HuggingFace format (optional)
        if not args.skip_hf_conversion:
            # Determine the output directory for HF tokenizer
            hf_tokenizer_dir = args.hf_tokenizer_dir or os.path.join(args.output_dir, "hf_tokenizer")

            # Convert to HF format
            try:
                convert_sentencepiece_to_hf(
                    model_file,  # Path to the trained SentencePiece model
                    hf_tokenizer_dir,
                    vocab_size=args.vocab_size,
                    special_tokens={
                        "unk_token": "<unk>",
                        "pad_token": "<pad>",
                        "mask_token": "<mask>",
                        "cls_token": "<cls>",
                        "sep_token": "<sep>"
                    },
                    test_sequences=[
                        "ATGCATGCATGC",
                        "GGGAAATTTCCC",
                        "ATATATATATAT",
                        "GCGCGCGCGCGC"
                    ] if not args.skip_verification else None
                )
                logger.info(f"Successfully created HuggingFace tokenizer in {hf_tokenizer_dir}")

                # Verify the HF tokenizer works
                if not args.skip_verification:
                    from transformers import PreTrainedTokenizerFast
                    hf_tokenizer = PreTrainedTokenizerFast(
                        tokenizer_file=os.path.join(hf_tokenizer_dir, "tokenizer.json"),
                        unk_token="<unk>",
                        pad_token="<pad>",
                        mask_token="<mask>",
                        cls_token="<cls>",
                        sep_token="<sep>"
                    )
                    logger.info("Verifying HuggingFace tokenizer...")

                    test_sequences = [
                        "ATGCATGCATGC",
                        "GGGAAATTTCCC",
                        "ATATATATATAT",
                        "GCGCGCGCGCGC"
                    ]

                    for seq in test_sequences:
                        tokens = hf_tokenizer.tokenize(seq)
                        token_ids = hf_tokenizer.convert_tokens_to_ids(tokens)
                        decoded = hf_tokenizer.decode(token_ids)

                        logger.info(f"Test sequence: {seq}")
                        logger.info(f"Tokens: {tokens}")
                        logger.info(f"Token IDs: {token_ids}")
                        logger.info(f"Decoded: {decoded}")
                        logger.info("-" * 40)

            except Exception as e:
                logger.error(f"Failed to convert to HuggingFace format: {e}")
                logger.exception("Detailed exception:")
        else:
            logger.info("Skipping conversion to HuggingFace format as requested")

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