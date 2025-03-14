"""
Genomic tokenizer implementation using SentencePiece for DNA sequences.
This module provides a tokenizer compatible with the HuggingFace ecosystem.
"""

import os
import logging
import shutil
from typing import List, Optional, Dict, Union, Tuple

import torch
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)

class GenomicTokenizer(PreTrainedTokenizer):
    """
    Genomic tokenizer based on SentencePiece for DNA sequences,
    compatible with HuggingFace's ecosystem.
    """
    vocab_files_names = {"spm_file": "spiece.model"}
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
            self,
            vocab_file=None,
            tokenizer_file=None,
            sentencepiece_model_file=None,
            do_lower_case=False,
            unk_token="<unk>",
            sep_token="<sep>",
            pad_token="<pad>",
            cls_token="<cls>",
            mask_token="<mask>",
            **kwargs
    ):
        # Initialize SentencePiece if model file is provided
        self.sentencepiece_model_file = sentencepiece_model_file or vocab_file

        if self.sentencepiece_model_file is not None:
            try:
                import sentencepiece as spm
                self.sp_model = spm.SentencePieceProcessor()
                logger.info(f"Loading SentencePiece model from {self.sentencepiece_model_file}")
                self.sp_model.Load(self.sentencepiece_model_file)
                logger.info(f"Loaded SentencePiece model with vocab size: {self.sp_model.GetPieceSize()}")
            except ImportError:
                raise ImportError("You need to install sentencepiece to use GenomicTokenizer: "
                                 "https://github.com/google/sentencepiece")
            except Exception as e:
                logger.error(f"Failed to load SentencePiece model: {e}")
                raise

        # Initialize parent class
        super().__init__(
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            do_lower_case=do_lower_case,
            **kwargs
        )

    def tokenize_with_safety(self, text, max_length=None, truncation=True, **kwargs):
        """Tokenize with explicit length safeguards to prevent indexing errors."""
        # Filter out invalid characters for genomic data
        if text:
            text = text.upper()
            text = ''.join(c for c in text if c in 'ATGC')

        # Check if text is empty after filtering
        if not text:
            return {"input_ids": torch.zeros(0, dtype=torch.long),
                    "attention_mask": torch.zeros(0, dtype=torch.long)}

        # Apply length limit if needed
        if max_length and len(text) > max_length and truncation:
            logger.warning(f"Truncating sequence from {len(text)} to {max_length}")
            text = text[:max_length]

        # Proceed with normal tokenization
        encoding = self.tokenize(text, **kwargs)
        return encoding

    @property
    def vocab_size(self):
        """Return the size of vocabulary."""
        if hasattr(self, "sp_model"):
            return self.sp_model.GetPieceSize()
        return 0

    def get_vocab(self):
        """Return the vocabulary as a dict."""
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, text):
        """Tokenize using SentencePiece model with proper empty string handling."""
        # Handle empty input
        if not text:
            return []

        # Ensure sequence is uppercase for DNA
        text = text.upper()
        # Filter out invalid characters
        text = ''.join(c for c in text if c in 'ATGC')

        # Check if text is empty after filtering
        if not text:
            return []

        # Tokenize with SentencePiece
        if hasattr(self, "sp_model"):
            tokens = self.sp_model.EncodeAsPieces(text)
            return tokens
        else:
            logger.error("SentencePiece model not initialized")
            return []

    def _convert_token_to_id(self, token):
        """Convert token to id using SentencePiece model with proper OOV handling.

        Args:
            token: The token to convert

        Returns:
            int: The token ID, or unk_token_id if token is out of vocabulary
        """
        if not hasattr(self, "sp_model"):
            logger.error("SentencePiece model not initialized")
            return self.unk_token_id

        # Get the token ID from SentencePiece
        token_id = self.sp_model.PieceToId(token)

        # SentencePiece returns -1 for unknown tokens or may return IDs beyond vocab size
        if token_id == -1 or token_id >= self.sp_model.GetPieceSize():
            logger.debug(f"OOV token encountered: '{token}', mapping to unk_token_id")
            return self.unk_token_id

        return token_id

    def _convert_id_to_token(self, index):
        """Convert id to token using SentencePiece model"""
        if hasattr(self, "sp_model"):
            token = self.sp_model.IdToPiece(index)
            return token
        else:
            logger.error("SentencePiece model not initialized")
            return self.unk_token

    def convert_tokens_to_string(self, tokens):
        """Convert tokens back to string"""
        if hasattr(self, "sp_model"):
            return self.sp_model.DecodePieces(tokens)
        else:
            return " ".join(tokens)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """Loading from pretrained with proper error handling"""
        instance = super().from_pretrained(*args, **kwargs)

        # Handle SentencePiece model loading
        spm_file = kwargs.get("spm_file", None) or getattr(instance, "sentencepiece_model_file", None)

        # If this is a directory, look for the SentencePiece model
        if os.path.isdir(args[0]):
            potential_spm_files = [
                os.path.join(args[0], "spiece.model"),
                os.path.join(args[0], "tokenizer.model"),
                os.path.join(args[0], "model.model"),
            ]
            for file_path in potential_spm_files:
                if os.path.exists(file_path):
                    spm_file = file_path
                    break

        if spm_file and os.path.exists(spm_file):
            import sentencepiece as spm
            instance.sentencepiece_model_file = spm_file
            instance.sp_model = spm.SentencePieceProcessor()
            instance.sp_model.Load(spm_file)
            logger.info(
                f"Loaded SentencePiece model from {spm_file} with vocab size {instance.sp_model.GetPieceSize()}")
        else:
            logger.warning(f"No SentencePiece model file found at {spm_file}")

        return instance

    def save_vocabulary(self, save_directory, filename_prefix=None):
        """
        Save the SentencePiece vocabulary file to the specified directory.

        Args:
            save_directory (str): Directory to save the vocabulary file
            filename_prefix (str, optional): Prefix to add to the filenames

        Returns:
            Tuple(str): Paths to the saved files
        """
        if not os.path.isdir(save_directory):
            os.makedirs(save_directory, exist_ok=True)

        if not hasattr(self, "sentencepiece_model_file") or not self.sentencepiece_model_file:
            logger.warning("No SentencePiece model file to save")
            return ()

        # Determine the save path
        out_name = "spiece.model"
        if filename_prefix is not None:
            out_name = f"{filename_prefix}-{out_name}"
        output_model_file = os.path.join(save_directory, out_name)

        # Copy the SentencePiece model file
        try:
            shutil.copyfile(self.sentencepiece_model_file, output_model_file)
            logger.info(f"SentencePiece model saved to {output_model_file}")
            return (output_model_file,)
        except Exception as e:
            logger.error(f"Error saving SentencePiece model: {e}")
            return ()

    def save_pretrained(self, save_directory, legacy_format=None, filename_prefix=None, push_to_hub=False, **kwargs):
        """Save tokenizer with proper handling of SentencePiece model."""
        if not os.path.exists(save_directory):
            os.makedirs(save_directory, exist_ok=True)

        # Save using the parent class method first
        saved_files = super().save_pretrained(
            save_directory,
            legacy_format=legacy_format,
            filename_prefix=filename_prefix,
            push_to_hub=push_to_hub,
            **kwargs
        )

        # Convert tuple to list so we can append
        saved_files_list = list(saved_files)

        # Additionally save the SentencePiece model file if available
        if hasattr(self, "sentencepiece_model_file") and self.sentencepiece_model_file:
            try:
                spm_dest = os.path.join(save_directory, "spiece.model")
                shutil.copyfile(self.sentencepiece_model_file, spm_dest)
                saved_files_list.append(spm_dest)
                logger.info(f"Copied SentencePiece model to {spm_dest}")
            except Exception as e:
                logger.warning(f"Failed to copy SentencePiece model: {e}")

        return tuple(saved_files_list)  # Convert back to tuple

    # Add genomic-specific methods
    def tokenize_genomic_sequence(self, sequence: str) -> List[str]:
        """Tokenize a DNA sequence with validation."""
        # Validate sequence
        sequence = sequence.upper()
        invalid_chars = set(sequence) - set('ATGC')
        if invalid_chars:
            logger.warning(f"Sequence contains invalid characters: {invalid_chars}")
            # Filter out non-ATGC characters
            sequence = ''.join(c for c in sequence if c in 'ATGC')

        return self.tokenize(sequence)


def train_sentencepiece_bpe(input_file: str, model_prefix: str, vocab_size: int = 4096,
                            character_coverage: float = 1.0, model_type: str = "bpe"):
    """Train a SentencePiece BPE tokenizer on genomic data with proper error handling."""
    import sentencepiece as spm
    import os
    import logging
    logger = logging.getLogger(__name__)

    # Verify input file exists and is not empty
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file for tokenizer training not found: {input_file}")

    file_size = os.path.getsize(input_file)
    if file_size == 0:
        raise ValueError(f"Input file for tokenizer training is empty: {input_file}")

    # SentencePiece training parameters
    training_args = {
        'input': input_file,
        'model_prefix': model_prefix,
        'vocab_size': vocab_size,
        'character_coverage': character_coverage,
        'model_type': model_type,
        'input_sentence_size': 10000000,  # Use a large number to process all sequences
        'shuffle_input_sentence': True,
        'normalization_rule_name': 'identity',  # No normalization for DNA
        'byte_fallback': False,  # DNA only has 4 characters, no need for byte fallback
        'unk_id': 0,
        'bos_id': 1,
        'eos_id': 2,
        'pad_id': 3,
    }

    try:
        # Train the tokenizer
        spm.SentencePieceTrainer.train(**training_args)
        model_file = f"{model_prefix}.model"
        vocab_file = f"{model_prefix}.vocab"

        # Verify the files were created
        if not os.path.exists(model_file) or not os.path.exists(vocab_file):
            raise FileNotFoundError(f"SentencePiece training failed to create model or vocab file")

        logger.info(f"Trained SentencePiece BPE tokenizer with vocab size {vocab_size}")
        return model_file, vocab_file

    except Exception as e:
        logger.error(f"SentencePiece training failed: {str(e)}")
        raise RuntimeError(f"Failed to train SentencePiece tokenizer: {str(e)}")


def prepare_genomic_data(input_files: List[str], output_file: str, sample_size: Optional[int] = None):
    """Prepare genomic data for tokenizer training by sampling from input files."""
    all_sequences = []

    for file_path in input_files:
        with open(file_path, 'r') as f:
            current_seq = ""
            for line in f:
                line = line.strip()
                # Skip FASTA headers
                if line.startswith('>'):
                    if current_seq:
                        all_sequences.append(current_seq)
                        current_seq = ""
                    continue
                # Only accept A, T, G, C sequences, ignore others
                if set(line.upper()) <= set('ATGC'):
                    current_seq += line.upper()

            # Add the last sequence if exists
            if current_seq:
                all_sequences.append(current_seq)

    # Sample if needed
    if sample_size and sample_size < len(all_sequences):
        import random
        all_sequences = random.sample(all_sequences, sample_size)

    # Write to output file - one sequence per line
    with open(output_file, 'w') as f:
        for seq in all_sequences:
            # Split long sequences into manageable chunks (e.g., 1000 bp)
            for i in range(0, len(seq), 1000):
                chunk = seq[i:i+1000]
                if len(chunk) > 50:  # Only keep reasonably sized chunks
                    f.write(chunk + '\n')

    logger.info(f"Prepared {len(all_sequences)} sequences for tokenizer training")
    return output_file


def create_genomic_tokenizer_from_model(model_path, **kwargs):
    """
    Create a genomic tokenizer directly from a SentencePiece model file.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"SentencePiece model file not found: {model_path}")

    return GenomicTokenizer(sentencepiece_model_file=model_path, **kwargs)


def setup_tokenizer(args, accelerator):
    """
    Set up tokenizer with improved reuse logic and proper special token handling.
    """
    import os
    import logging
    logger = logging.getLogger(__name__)

    # Paths for different tokenizer components
    output_tokenizer_dir = os.path.join(args.output_dir, "tokenizer")
    spm_model_path = os.path.join(args.output_dir, "genomic_bpe.model")
    direct_spm_path = os.path.join(output_tokenizer_dir, "spiece.model")

    # Create tokenizer directory if it doesn't exist
    if accelerator.is_main_process:
        os.makedirs(output_tokenizer_dir, exist_ok=True)

    # Synchronize to ensure directory exists for all processes
    accelerator.wait_for_everyone()

    # PRIORITY 1: Try user-provided tokenizer path
    if args.tokenizer_path and os.path.exists(args.tokenizer_path):
        logger.info(f"Loading tokenizer from user-provided path: {args.tokenizer_path}")
        try:
            # Load tokenizer normally but ensure special tokens are properly configured
            tokenizer = GenomicTokenizer.from_pretrained(args.tokenizer_path)
            logger.info(f"Successfully loaded tokenizer with vocab size: {tokenizer.vocab_size}")

            # Ensure mask token exists and is properly defined
            _ensure_special_tokens(tokenizer)
            return tokenizer
        except Exception as e:
            logger.warning(f"Failed to load tokenizer from {args.tokenizer_path}: {e}")
            # Continue to next option instead of failing

    # PRIORITY 2: Try saved tokenizer directory
    if os.path.exists(output_tokenizer_dir):
        logger.info(f"Trying to load tokenizer from previous run: {output_tokenizer_dir}")
        try:
            tokenizer = GenomicTokenizer.from_pretrained(output_tokenizer_dir)
            logger.info(f"Successfully loaded saved tokenizer with vocab size: {tokenizer.vocab_size}")

            # Ensure mask token exists and is properly defined
            _ensure_special_tokens(tokenizer)
            return tokenizer
        except Exception as e:
            logger.warning(f"Failed to load saved tokenizer: {e}")
            # Continue to next option

    # PRIORITY 3: Try to load SentencePiece model directly
    for spm_path in [direct_spm_path, spm_model_path]:
        if os.path.exists(spm_path):
            logger.info(f"Trying to load SentencePiece model directly: {spm_path}")
            try:
                # Create tokenizer with necessary special tokens
                tokenizer = GenomicTokenizer(
                    sentencepiece_model_file=spm_path,
                    # Include all necessary special tokens
                    unk_token="<unk>",
                    pad_token="<pad>",
                    mask_token="<mask>"
                )

                logger.info(f"Successfully loaded SentencePiece model with vocab size: {tokenizer.vocab_size}")

                # Ensure mask token exists and is properly defined
                _ensure_special_tokens(tokenizer)

                # Save in standard format for future runs
                if accelerator.is_main_process:
                    tokenizer.save_pretrained(output_tokenizer_dir)
                    logger.info(f"Saved loaded tokenizer to {output_tokenizer_dir}")

                accelerator.wait_for_everyone()
                return tokenizer
            except Exception as e:
                logger.warning(f"Failed to load SentencePiece model {spm_path}: {e}")
                # Continue to next option

    # PRIORITY 4: No tokenizer found, train a new one
    if accelerator.is_main_process:
        logger.info("No existing tokenizer found. Training new BPE tokenizer on genomic data")
        try:
            # Prepare data for tokenizer training
            prepared_data = prepare_genomic_data(
                args.input_files,
                os.path.join(args.output_dir, "prepared_sequences.txt"),
                args.tokenizer_sample_size
            )

            # Train SentencePiece BPE tokenizer
            model_prefix = os.path.join(args.output_dir, "genomic_bpe")
            model_file, vocab_file = train_sentencepiece_bpe(
                prepared_data,
                model_prefix,
                args.vocab_size
            )

            # Initialize tokenizer with required special tokens
            tokenizer = GenomicTokenizer(
                sentencepiece_model_file=model_file,
                unk_token="<unk>",
                pad_token="<pad>",
                mask_token="<mask>"
            )

            # Ensure mask token exists and is properly defined
            _ensure_special_tokens(tokenizer)

            # Save the tokenizer to the specified path
            tokenizer.save_pretrained(output_tokenizer_dir)
            logger.info(f"Trained new tokenizer and saved to {output_tokenizer_dir}")
        except Exception as e:
            logger.error(f"Error creating tokenizer: {str(e)}")
            raise RuntimeError(f"Failed to create tokenizer: {str(e)}")

    # Wait for tokenizer to be saved by main process
    accelerator.wait_for_everyone()

    # Non-main processes load the saved tokenizer
    if not accelerator.is_main_process:
        logger.info(f"Loading tokenizer from {output_tokenizer_dir}")
        try:
            tokenizer = GenomicTokenizer.from_pretrained(output_tokenizer_dir)
            # Ensure mask token exists and is properly defined
            _ensure_special_tokens(tokenizer)
        except Exception as e:
            raise RuntimeError(f"Failed to load created tokenizer: {str(e)}")

    # Verify tokenizer is properly loaded on all processes
    vocab_size = tokenizer.vocab_size
    logger.info(f"Tokenizer ready with vocabulary size: {vocab_size}")

    # Log special tokens map for debugging
    logger.info("Special tokens configuration:")
    if hasattr(tokenizer, 'special_tokens_map'):
        for name, token in tokenizer.special_tokens_map.items():
            if token is not None:
                token_id = tokenizer.convert_tokens_to_ids(token) if token else None
                logger.info(f"  {name}: {token} -> ID: {token_id}")

    return tokenizer


# In tokenization/genomic_tokenizer.py, modify the _ensure_special_tokens function
def _ensure_special_tokens(tokenizer):
    # Get vocabulary size for validation
    vocab_size = tokenizer.vocab_size

    # Fix mask_token_id directly if it exceeds vocab size
    if hasattr(tokenizer, 'mask_token_id') and tokenizer.mask_token_id is not None:
        if tokenizer.mask_token_id >= vocab_size:
            # Use a fixed safe ID
            safe_id = 1  # Use a simple, safe ID within vocab range

            # Store the original token text
            mask_token_text = tokenizer.mask_token

            # Update the tokenizer's internal mappings
            tokenizer.mask_token_id = safe_id

            # Update the special tokens dictionary if it exists
            if hasattr(tokenizer, 'special_tokens_map'):
                tokenizer.special_tokens_map['mask_token'] = mask_token_text

            # Make sure the vocab knows about this mapping
            if hasattr(tokenizer, 'added_tokens_encoder'):
                tokenizer.added_tokens_encoder[mask_token_text] = safe_id
            if hasattr(tokenizer, 'added_tokens_decoder'):
                tokenizer.added_tokens_decoder[safe_id] = mask_token_text

            logger.info(f"Fixed mask_token to use ID {safe_id} instead of {tokenizer.mask_token_id}")