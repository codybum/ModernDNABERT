#!/usr/bin/env python
"""
Standalone utility to convert SentencePiece models to HuggingFace tokenizers format.
This script allows you to convert existing SentencePiece models to a format compatible
with HuggingFace's transformers library.

Usage:
    python convert_sp_to_hf.py --spm_model path/to/spiece.model --output path/to/output_dir
"""

import os
import glob
import argparse
import logging
import json
from typing import Dict, Optional, List

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


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


def main():
    """Main function for the conversion script."""
    parser = argparse.ArgumentParser(
        description="Convert SentencePiece model to HuggingFace tokenizers format"
    )

    # Required arguments
    parser.add_argument("--spm_model", required=True, type=str,
                        help="Path to the SentencePiece model file or directory containing it")
    parser.add_argument("--output", required=True, type=str,
                        help="Output directory for the converted tokenizer")

    # Optional arguments
    parser.add_argument("--vocab_size", type=int, default=None,
                        help="Override vocabulary size (default: use model's size)")
    parser.add_argument("--unk_token", type=str, default="<unk>",
                        help="Unknown token (default: <unk>)")
    parser.add_argument("--pad_token", type=str, default="<pad>",
                        help="Padding token (default: <pad>)")
    parser.add_argument("--mask_token", type=str, default="<mask>",
                        help="Mask token (default: <mask>)")
    parser.add_argument("--cls_token", type=str, default="<cls>",
                        help="Classification token (default: <cls>)")
    parser.add_argument("--sep_token", type=str, default="<sep>",
                        help="Separator token (default: <sep>)")
    parser.add_argument("--test", action="store_true",
                        help="Test the converted tokenizer")

    args = parser.parse_args()

    # Verify input path exists
    if not os.path.exists(args.spm_model):
        logger.error(f"Path not found: {args.spm_model}")
        return 1

    # Create special tokens dictionary
    special_tokens = {
        "unk_token": args.unk_token,
        "pad_token": args.pad_token,
        "mask_token": args.mask_token,
        "cls_token": args.cls_token,
        "sep_token": args.sep_token
    }

    # Test sequences for genomic data if testing is enabled
    test_sequences = None
    if args.test:
        test_sequences = [
            "ATGCATGCATGC",
            "GGGAAATTTCCC",
            "ATATATATATAT",
            "GCGCGCGCGCGC"
        ]

    try:
        # Perform the conversion
        convert_sentencepiece_to_hf(
            args.spm_model,
            args.output,
            args.vocab_size,
            special_tokens,
            test_sequences
        )
        logger.info(f"Successfully converted {args.spm_model} to HuggingFace format in {args.output}")

        # Show usage example
        logger.info("\nUsage example in Python:")
        logger.info("from transformers import PreTrainedTokenizerFast")
        logger.info(f"tokenizer = PreTrainedTokenizerFast.from_pretrained('{args.output}')")
        logger.info("# Or with explicit parameters:")
        logger.info(f"tokenizer = PreTrainedTokenizerFast(")
        logger.info(f"    tokenizer_file='{args.output}/tokenizer.json',")
        logger.info(f"    unk_token='{args.unk_token}',")
        logger.info(f"    pad_token='{args.pad_token}',")
        logger.info(f"    mask_token='{args.mask_token}',")
        logger.info(f"    cls_token='{args.cls_token}',")
        logger.info(f"    sep_token='{args.sep_token}'")
        logger.info(f")")

        return 0
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())