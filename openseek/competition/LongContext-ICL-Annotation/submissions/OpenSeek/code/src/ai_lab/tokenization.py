from typing import Optional


class TokenCounter:
    def __init__(self, tokenizer_path: str, trust_remote_code: bool = True) -> None:
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=trust_remote_code,
        )

    def count(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))


def maybe_build_token_counter(
    tokenizer_path: str, trust_remote_code: bool = True
) -> Optional[TokenCounter]:
    try:
        return TokenCounter(tokenizer_path, trust_remote_code=trust_remote_code)
    except Exception:
        return None
