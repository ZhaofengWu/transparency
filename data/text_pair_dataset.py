import argparse
import os
from typing import Any, Union

from transformers import PreTrainedTokenizerBase

from paoding import LocalDataset

NUM_PRETRAINING_EMBEDDINGS = 1200


class TextPairDataset(LocalDataset):
    def __init__(
        self,
        hparams: argparse.Namespace,
        tokenizer: PreTrainedTokenizerBase,
        preprocess_and_save=True,
    ):
        if hparams.data_type == "nl":
            assert hparams.tokenizer == "pretrained"
            data_format = "json"
            kwargs = {}
        else:
            data_format = "csv"
            kwargs = {
                "delimiter": "\t",
                "column_names": [self.text_key, self.second_text_key, self.label_key],
            }

        super().__init__(
            hparams,
            tokenizer,
            lambda split: f"{hparams.data_type}.{split}",
            data_format,
            preprocess_and_save=preprocess_and_save,
            tokenize_separately=True,
            **kwargs,
        )

    @property
    def hash_fields(self) -> list[Any]:
        return super().hash_fields + [
            self.hparams.data_type,
            self.hparams.tokenizer,
            self.hparams.model_name_or_path,
        ]

    @property
    def test_splits(self) -> list[str]:
        if self.hparams.data_type != "nl":
            return super().test_splits
        else:
            files = sorted(os.listdir(self.hparams.data_dir))
            return [file[len("nl.") :] for file in files if file.startswith("nl.test_")]

    @property
    def sort_key(self) -> Union[str, tuple[str]]:
        return ("input_ids_1", "input_ids_2")

    @property
    def second_text_key(self) -> str:
        return "second_text"

    @property
    def metric_names(self) -> list[str]:
        return ["boolean_accuracy"]

    @property
    def metric_watch_mode(self) -> str:
        return "max"

    @property
    def output_mode(self) -> str:
        return "classification"

    @property
    def num_labels(self) -> int:
        return 2

    @property
    def tokenize_kwargs(self) -> dict[str, Any]:
        if self.hparams.tokenizer == "pretrained":
            return dict(padding=False, truncation=False)
        else:
            assert self.hparams.tokenizer == "custom"
            return {}

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        LocalDataset.add_args(parser)
        parser.add_argument("--data_type", required=True, type=str, choices=["pl", "nl"])
