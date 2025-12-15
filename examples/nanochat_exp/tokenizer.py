"""
Tokenizer wrapper using HuggingFace tokenizers library (easy to install alternative to rustbpe).

This module provides a tokenizer interface compatible with nanochat's tokenizer API,
but uses HuggingFace's tokenizers library instead of rustbpe, which requires Rust compilation.
"""

import os
from typing import List, Optional, Union, Dict

try:
    from tokenizers import Tokenizer as HFTokenizer
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
except ImportError:
    raise ImportError(
        "tokenizers library is required. Please install it with: pip install tokenizers\n"
        "This is much easier than installing rustbpe (no Rust compilation needed)."
    )


class TokenizerWrapper:
    """
    Wrapper around HuggingFace Tokenizer to match nanochat's tokenizer interface.
    
    This provides a drop-in replacement for nanochat's rustbpe-based tokenizer,
    using the easier-to-install HuggingFace tokenizers library.
    """
    
    def __init__(self, tokenizer_path: Optional[str] = None):
        """
        Initialize the tokenizer.
        
        Args:
            tokenizer_path: Path to tokenizer.json file. If None, tries to find it
                          in standard nanochat locations or creates a default one.
        """
        self.tokenizer = None
        self.bos_token_id = None
        self.eos_token_id = None
        
        # Try to load existing tokenizer
        if tokenizer_path and os.path.exists(tokenizer_path):
            self._load_tokenizer(tokenizer_path)
        else:
            # Try to find tokenizer in standard locations
            tokenizer_path = self._find_tokenizer()
            if tokenizer_path:
                self._load_tokenizer(tokenizer_path)
            else:
                # Create a default tokenizer (fallback)
                self._create_default_tokenizer()
    
    def _find_tokenizer(self) -> Optional[str]:
        """Try to find tokenizer.json in standard nanochat locations."""
        possible_paths = [
            # Check OPENSEEK_NANOCHAT_DATA_DIR (highest priority)
            os.path.join(os.environ.get("OPENSEEK_NANOCHAT_DATA_DIR", ""), "tokenizer", "tokenizer.json"),
            # Check NANOCHAT_BASE_DIR
            os.path.join(os.environ.get("NANOCHAT_BASE_DIR", ""), "tokenizer", "tokenizer.json"),
            # Check common nanochat locations
            os.path.join(os.path.expanduser("~"), ".cache", "openseek_nanochat", "tokenizer", "tokenizer.json"),
            os.path.join(os.path.expanduser("~"), ".cache", "nanochat", "tokenizer", "tokenizer.json"),
            # Check relative to this file
            os.path.join(os.path.dirname(__file__), "../../../tokenizer", "tokenizer.json"),
        ]
        
        for path in possible_paths:
            if path and os.path.exists(path):
                return path
        
        return None
    
    def _load_tokenizer(self, tokenizer_path: str):
        """Load tokenizer from tokenizer.json file."""
        try:
            self.tokenizer = HFTokenizer.from_file(tokenizer_path)
            # Try to get BOS/EOS tokens from the tokenizer
            # HuggingFace tokenizers typically use special tokens
            try:
                # Try to get special tokens
                if hasattr(self.tokenizer, 'token_to_id'):
                    # Common BOS/EOS token IDs
                    self.bos_token_id = self.tokenizer.token_to_id("<|bos|>") or \
                                       self.tokenizer.token_to_id("<|begin_of_text|>") or \
                                       self.tokenizer.token_to_id("[BOS]") or \
                                       self.tokenizer.token_to_id("<s>") or 1
                    self.eos_token_id = self.tokenizer.token_to_id("<|eos|>") or \
                                       self.tokenizer.token_to_id("<|end_of_text|>") or \
                                       self.tokenizer.token_to_id("[EOS]") or \
                                       self.tokenizer.token_to_id("</s>") or 2
                else:
                    # Fallback
                    self.bos_token_id = 1
                    self.eos_token_id = 2
            except:
                self.bos_token_id = 1
                self.eos_token_id = 2
        except Exception as e:
            raise RuntimeError(f"Failed to load tokenizer from {tokenizer_path}: {e}")
    
    def _create_default_tokenizer(self):
        """Create a default BPE tokenizer as fallback."""
        # This is a minimal tokenizer - in practice, you should train or load a proper one
        self.tokenizer = HFTokenizer(BPE())
        self.tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        self.bos_token_id = 1
        self.eos_token_id = 2
        print("Warning: Using default tokenizer. For best results, use a trained tokenizer.")
    
    def get_bos_token_id(self) -> int:
        """Get the BOS (beginning of sequence) token ID."""
        return self.bos_token_id if self.bos_token_id is not None else 1
    
    def get_eos_token_id(self) -> int:
        """Get the EOS (end of sequence) token ID."""
        return self.eos_token_id if self.eos_token_id is not None else 2
    
    def token_to_id(self, token_str: str) -> Optional[int]:
        """
        Convert a token string to its ID.
        
        Args:
            token_str: Token string to convert
        
        Returns:
            Token ID, or None if not found
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not initialized")
        
        try:
            if hasattr(self.tokenizer, 'token_to_id'):
                return self.tokenizer.token_to_id(token_str)
            return None
        except:
            return None
    
    def id_to_token(self, token_id: int) -> Optional[str]:
        """
        Convert a token ID to its string representation.
        
        Args:
            token_id: Token ID to convert
        
        Returns:
            Token string, or None if not found
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not initialized")
        
        try:
            if hasattr(self.tokenizer, 'id_to_token'):
                return self.tokenizer.id_to_token(token_id)
            # Fallback: try decoding
            return self.tokenizer.decode([token_id], skip_special_tokens=False)
        except:
            return None
    
    def get_vocab_size(self) -> int:
        """Get the vocabulary size."""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not initialized")
        
        try:
            # Method 1: Try to get vocab size directly from tokenizer
            if hasattr(self.tokenizer, 'get_vocab_size'):
                vocab_size = self.tokenizer.get_vocab_size()
                if vocab_size and vocab_size > 4:  # Sanity check
                    return vocab_size
            
            # Method 2: Try to get from model (most reliable for BPE)
            if hasattr(self.tokenizer, 'model'):
                model = self.tokenizer.model
                if hasattr(model, 'get_vocab_size'):
                    vocab_size = model.get_vocab_size()
                    if vocab_size and vocab_size > 4:
                        return vocab_size
                # For BPE model, check vocab directly
                if hasattr(model, 'vocab') and model.vocab:
                    vocab_size = len(model.vocab)
                    if vocab_size > 4:
                        return vocab_size
                # Check merges for BPE (vocab_size = base_vocab_size + num_merges)
                if hasattr(model, 'merges') and model.merges:
                    # BPE vocab size = 256 (base bytes) + len(merges) + special tokens
                    base_size = 256  # Byte-level base vocabulary
                    merges_size = len(model.merges)
                    special_tokens_size = 4  # BOS, EOS, PAD, UNK
                    vocab_size = base_size + merges_size + special_tokens_size
                    if vocab_size > 4:
                        return vocab_size
            
            # Method 3: Try to get from vocab dict (may only contain special tokens)
            if hasattr(self.tokenizer, 'get_vocab'):
                vocab = self.tokenizer.get_vocab()
                if vocab:
                    vocab_size = len(vocab)
                    # Check if vocab only contains special tokens (size <= 10)
                    # If so, try to get actual vocab size from model
                    if vocab_size <= 10:
                        # Likely only special tokens, try model instead
                        if hasattr(self.tokenizer, 'model') and hasattr(self.tokenizer.model, 'vocab'):
                            model_vocab = self.tokenizer.model.vocab
                            if model_vocab and len(model_vocab) > vocab_size:
                                return len(model_vocab)
                    return vocab_size
            
            # Method 4: Try to infer from max token ID in vocab
            if hasattr(self.tokenizer, 'get_vocab'):
                vocab = self.tokenizer.get_vocab()
                if vocab:
                    max_token_id = max(vocab.values()) if vocab.values() else 0
                    # Vocab size should be at least max_token_id + 1
                    if max_token_id > 0:
                        inferred_size = max_token_id + 1
                        if inferred_size > 4:
                            return inferred_size
            
            # Method 5: Try common vocabulary sizes by testing decode
            for vocab_size_guess in [50257, 32000, 50000, 65536, 256]:
                try:
                    # Try to decode a token near the end of vocabulary
                    test_token_id = vocab_size_guess - 1
                    test_token = self.tokenizer.decode([test_token_id], skip_special_tokens=False)
                    if test_token:  # If successful, this might be the vocab size
                        return vocab_size_guess
                except:
                    continue
            
            # Last resort: return a default
            print("Warning: Could not determine vocabulary size, using default 50257")
            print("This may indicate the tokenizer was not properly trained or loaded.")
            print("Please ensure you have trained a tokenizer using: python -m examples.nanochat_exp.tok_train")
            return 50257  # Common GPT-2 style vocab size
        except Exception as e:
            # Fallback to default
            print(f"Warning: Error getting vocabulary size: {e}")
            print("Using default vocabulary size: 50257")
            return 50257
    
    def encode(
        self,
        texts: Union[str, List[str]],
        prepend: Optional[int] = None,
        num_threads: int = 4
    ) -> Union[List[int], List[List[int]]]:
        """
        Encode text(s) into token IDs.
        
        Args:
            texts: Single text string or list of text strings
            prepend: Token ID to prepend to each sequence (e.g., BOS token)
            num_threads: Number of threads (for compatibility, HF tokenizers handles this internally)
        
        Returns:
            If single text: list of token IDs
            If list of texts: list of lists of token IDs
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not initialized")
        
        is_single = isinstance(texts, str)
        if is_single:
            texts = [texts]
        
        # Encode texts
        encodings = self.tokenizer.encode_batch(texts, add_special_tokens=False)
        
        # Extract token IDs
        results = []
        for encoding in encodings:
            token_ids = encoding.ids
            # Prepend token if specified
            if prepend is not None:
                token_ids = [prepend] + token_ids
            results.append(token_ids)
        
        # Return single list if single text was provided
        if is_single:
            return results[0]
        return results
    
    def __getattr__(self, name: str):
        """
        Forward undefined attribute access to the underlying HuggingFace tokenizer.
        
        This allows access to any methods or attributes of the HuggingFace tokenizer
        that we haven't explicitly wrapped.
        """
        if self.tokenizer is None:
            raise AttributeError(f"Tokenizer not initialized, cannot access '{name}'")
        
        if hasattr(self.tokenizer, name):
            return getattr(self.tokenizer, name)
        
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    def decode(self, token_ids: Union[List[int], List[List[int]]]) -> Union[str, List[str]]:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: Single list of token IDs or list of lists
        
        Returns:
            Decoded text string or list of strings
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not initialized")
        
        is_single = isinstance(token_ids[0], int) if token_ids else True
        
        if is_single:
            token_ids = [token_ids]
        
        results = []
        for ids in token_ids:
            # Remove prepended BOS token if present
            if ids and ids[0] == self.get_bos_token_id():
                ids = ids[1:]
            text = self.tokenizer.decode(ids, skip_special_tokens=False)
            results.append(text)
        
        if is_single:
            return results[0]
        return results


# Global tokenizer instance
_tokenizer_instance: Optional[TokenizerWrapper] = None


def get_tokenizer(tokenizer_path: Optional[str] = None) -> TokenizerWrapper:
    """
    Get or create the global tokenizer instance.
    
    This function matches nanochat's get_tokenizer() interface.
    
    Args:
        tokenizer_path: Optional path to tokenizer.json file
    
    Returns:
        TokenizerWrapper instance
    """
    global _tokenizer_instance
    
    if _tokenizer_instance is None:
        _tokenizer_instance = TokenizerWrapper(tokenizer_path)
    
    return _tokenizer_instance


def reset_tokenizer():
    """Reset the global tokenizer instance (useful for testing)."""
    global _tokenizer_instance
    _tokenizer_instance = None


def get_token_bytes(token_id: Optional[int] = None, device: Optional[str] = None) -> Union[bytes, dict]:
    """
    Get the byte representation of token ID(s).
    
    This function is used by nanochat for various purposes like counting tokens,
    verifying tokenizer correctness, calculating bits per byte (bpb), etc.
    
    When called with device parameter (and no token_id), returns a dictionary
    mapping all token IDs to their byte representations, optionally moved to device.
    
    Args:
        token_id: Optional token ID to convert to bytes. If None, returns all tokens.
        device: Optional device string ("cuda", "cpu", etc.). If provided without token_id,
                returns a dict mapping all token IDs to bytes.
    
    Returns:
        If token_id is provided: bytes representation of the token
        If device is provided (without token_id): dict mapping token_id -> bytes
        Otherwise: dict mapping all token IDs to bytes
    """
    tokenizer_obj = get_tokenizer()
    
    if tokenizer_obj.tokenizer is None:
        raise RuntimeError("Tokenizer not initialized")
    
    # If device is provided without token_id, return all token bytes as dict
    if device is not None and token_id is None:
        return _get_all_token_bytes(tokenizer_obj, device)
    
    # If token_id is provided, return bytes for that token
    if token_id is not None:
        return _get_single_token_bytes(tokenizer_obj, token_id)
    
    # Default: return all token bytes as dict
    return _get_all_token_bytes(tokenizer_obj)


def _get_single_token_bytes(tokenizer_obj: TokenizerWrapper, token_id: int) -> bytes:
    """Get bytes for a single token ID."""
    try:
        # Decode single token ID
        text = tokenizer_obj.tokenizer.decode([token_id], skip_special_tokens=False)
        # Convert to bytes (UTF-8 encoding)
        return text.encode('utf-8')
    except Exception:
        # Fallback for special tokens or edge cases
        try:
            # Some tokenizers have id_to_token method
            if hasattr(tokenizer_obj.tokenizer, 'id_to_token'):
                token_str = tokenizer_obj.tokenizer.id_to_token(token_id)
                if token_str:
                    return token_str.encode('utf-8')
        except:
            pass
        
        # Handle common special tokens
        if token_id == tokenizer_obj.get_bos_token_id():
            return b"<|bos|>"
        elif token_id == tokenizer_obj.get_eos_token_id():
            return b"<|eos|>"
        else:
            # Last resort: try to decode as single character or return empty
            try:
                # Try treating as a single byte token (for byte-level tokenizers)
                if 0 <= token_id < 256:
                    return bytes([token_id])
            except:
                pass
            # Return empty bytes as last resort
            return b""


def _get_all_token_bytes(tokenizer_obj: TokenizerWrapper, device: Optional[str] = None) -> Dict[int, bytes]:
    """Get bytes for all tokens in the vocabulary."""
    try:
        # Try to get vocabulary from tokenizer
        vocab = None
        vocab_size = None
        
        # Method 1: Try get_vocab() which returns {token_str: token_id}
        if hasattr(tokenizer_obj.tokenizer, 'get_vocab'):
            vocab = tokenizer_obj.tokenizer.get_vocab()
            # Get actual vocab size from max token ID, not just dict length
            if vocab:
                max_token_id = max(vocab.values()) if vocab.values() else 0
                vocab_size = max(max_token_id + 1, len(vocab))
        
        # Method 2: Try get_vocab_size()
        if vocab_size is None and hasattr(tokenizer_obj.tokenizer, 'get_vocab_size'):
            vocab_size = tokenizer_obj.tokenizer.get_vocab_size()
        
        # Method 3: Try to get from model
        if vocab_size is None and hasattr(tokenizer_obj.tokenizer, 'model'):
            model = tokenizer_obj.tokenizer.model
            if hasattr(model, 'get_vocab_size'):
                vocab_size = model.get_vocab_size()
            elif hasattr(model, 'vocab') and model.vocab:
                vocab_size = len(model.vocab)
            elif hasattr(model, 'merges') and model.merges:
                # BPE vocab size = 256 (base bytes) + len(merges) + special tokens
                vocab_size = 256 + len(model.merges) + 4
        
        # Build mapping of token_id -> bytes
        token_bytes_dict: Dict[int, bytes] = {}
        
        if vocab is not None:
            # Use vocab dict directly (more efficient)
            for token_str, token_id in vocab.items():
                try:
                    token_bytes_dict[token_id] = token_str.encode('utf-8')
                except:
                    token_bytes_dict[token_id] = _get_single_token_bytes(tokenizer_obj, token_id)
            
            # Ensure all token IDs up to vocab_size are covered
            if vocab_size is not None:
                for token_id in range(vocab_size):
                    if token_id not in token_bytes_dict:
                        token_bytes_dict[token_id] = _get_single_token_bytes(tokenizer_obj, token_id)
        elif vocab_size is not None:
            # Iterate through token IDs
            for token_id in range(vocab_size):
                token_bytes_dict[token_id] = _get_single_token_bytes(tokenizer_obj, token_id)
        else:
            # Fallback: try common vocabulary sizes
            for vocab_size_guess in [50257, 32000, 50000, 65536]:
                try:
                    # Test if this vocab size is valid by trying to decode
                    test_token = tokenizer_obj.tokenizer.decode([vocab_size_guess - 1], skip_special_tokens=False)
                    vocab_size = vocab_size_guess
                    break
                except:
                    continue
            
            if vocab_size:
                for token_id in range(vocab_size):
                    token_bytes_dict[token_id] = _get_single_token_bytes(tokenizer_obj, token_id)
            else:
                # Last resort: return empty dict
                return {}
        
        # Create a wrapper dict that handles missing keys gracefully
        # This is important because nanochat's evaluate_bpb may access token IDs
        # that are outside the vocabulary (e.g., from model predictions)
        class TokenBytesDict(dict):
            """Dictionary that handles missing token IDs gracefully."""
            def __init__(self, base_dict, tokenizer_obj):
                super().__init__(base_dict)
                self.tokenizer_obj = tokenizer_obj
            
            def __getitem__(self, key):
                import torch
                
                # Handle tensor keys - nanochat passes entire tensors as keys
                if isinstance(key, torch.Tensor):
                    # If it's a tensor, we need to map each element
                    # Convert tensor to numpy/list and process each element
                    if key.numel() > 1:
                        # Multi-element tensor: return a tensor of byte lengths
                        # nanochat expects num_bytes2d which is a tensor of byte counts
                        device = key.device
                        dtype = key.dtype
                        
                        # Flatten the tensor and process each element
                        flat_key = key.flatten()
                        result_list = []
                        
                        for token_id in flat_key:
                            token_id_int = int(token_id.item())
                            if token_id_int in self:
                                bytes_val = super().__getitem__(token_id_int)
                            else:
                                # Try to decode even if not in vocab
                                try:
                                    bytes_val = _get_single_token_bytes(self.tokenizer_obj, token_id_int)
                                except:
                                    bytes_val = b""
                            
                            # Return length of bytes (nanochat expects byte counts)
                            result_list.append(len(bytes_val))
                        
                        # Reshape to match original tensor shape
                        result_tensor = torch.tensor(result_list, dtype=torch.long, device=device)
                        return result_tensor.view(key.shape)
                    else:
                        # Single element tensor
                        key = int(key.item())
                
                # Handle scalar keys
                if hasattr(key, '__int__'):
                    key = int(key)
                else:
                    key = int(key)
                
                if key in self:
                    return super().__getitem__(key)
                # Try to decode the token ID even if it's not in vocab
                try:
                    return _get_single_token_bytes(self.tokenizer_obj, key)
                except:
                    # Last resort: return empty bytes
                    return b""
        
        return TokenBytesDict(token_bytes_dict, tokenizer_obj)
    except Exception as e:
        print(f"Warning: Error building token_bytes dict: {e}")
        # Return empty dict - nanochat should handle this gracefully
        return {}
