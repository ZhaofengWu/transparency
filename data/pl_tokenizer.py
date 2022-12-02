from typing import Any

from paoding import Tokenizer

# Save the first 100 tokens for future use
TOKEN2ID = {"(": 100, ")": 101, "!": 102, "&": 103, "|": 104, "=": 105, "True": 106, "False": 107}
ID2TOKEN = {id: token for token, id in TOKEN2ID.items()}


class PlTokenizer(Tokenizer):
    def __len__(self) -> int:
        return max(TOKEN2ID.values()) + 1

    def _tokenize(
        self,
        text: str,
        add_special_tokens=False,
        truncation=False,
        max_length=None,
    ) -> dict[str, Any]:
        ids = []
        i = 0
        while i < len(text):
            c = text[i]
            if c in TOKEN2ID:
                ids.append(TOKEN2ID[c])
                i += 1
            elif c == "T":
                assert text[i + 1 : i + 4] == "rue"
                ids.append(TOKEN2ID["True"])
                i += 4
            elif c == "F":
                assert text[i + 1 : i + 5] == "alse"
                ids.append(TOKEN2ID["False"])
                i += 5
            else:
                assert False
        if max_length is not None:
            assert len(ids) <= max_length
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    def _convert_token_to_id(self, token: str) -> int:
        if token in self.special_token_to_id:
            return self.special_token_to_id[token]
        else:
            return TOKEN2ID[token]

    def _convert_id_to_token(self, id: int) -> str:
        if id in self.special_id_to_token:
            return self.special_id_to_token[id]
        else:
            return ID2TOKEN[id]

    def real_token_indices(self) -> list[int]:
        return list(TOKEN2ID.values())
