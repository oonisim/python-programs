#!/usr/bin/env python3
import argparse
import tiktoken
import json

class TiktokenAdapter:
    """Adapter for tiktoken to match a simple Tokenizer protocol."""

    def __init__(self, encoding_name: str):
        self._enc = tiktoken.get_encoding(encoding_name)
        self._eot_id = self._enc.eot_token
        #print(json.dumps(dir(self._enc), indent=2, default=str))

    @property
    def vocab_size(self) -> int:
        return self._enc.n_vocab

    @property
    def eot_token_id(self) -> int:
        # tiktoken exposes an "end of text" token; we treat it as EOS.
        return self._eot_id

    def encode(self, text: str) -> list[int]:
        return self._enc.encode(text)

    def special_tokens(self):
        return self._enc.special_tokens_set


def main() -> None:
    parser = argparse.ArgumentParser(description="Print tiktoken encoding token IDs.")
    parser.add_argument(
        "encodings",
        nargs="+",
        help="tiktoken encoding name(s), e.g. gpt2 cl100k_base o200k_base",
    )
    args = parser.parse_args()

    for name in args.encodings:
        tok = TiktokenAdapter(name)
        print(f"Model {name}")
        print(f"  vocab_size: {tok.vocab_size}")
        print(f"  EOT: {tok.eot_token_id}")
        print(f"  Special tokens: {tok.special_tokens()}")
        print()


if __name__ == "__main__":
    main()