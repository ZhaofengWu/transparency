from typing import Any

from allennlp.modules.matrix_attention import BilinearMatrixAttention
from allennlp.modules.scalar_mix import ScalarMix
import torch
from torch import nn
from transformers import AutoTokenizer
from transformers import PreTrainedTokenizerBase

from paoding import Model, Pooler, Transformer

from data.pl_tokenizer import PlTokenizer


class TransformerProbe(Model):
    def __init__(self, hparams):
        super().__init__(hparams)

        self.transformer = Transformer(self.hparams, "base", trainable=False)
        hidden_size = self.transformer.model.config.hidden_size

        if self.hparams.scalar_mix:
            config = self.transformer.model.config
            # +1 for embedding layer
            self.scalar_mix = ScalarMix(getattr(config, "n_layer", config.num_hidden_layers) + 1)

        if self.hparams.mlp_layers > 0:
            mlp = []
            for _ in range(self.hparams.mlp_layers):
                mlp.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU()])
            self.mlp = nn.Sequential(*mlp)

        self.pooler = Pooler(self.hparams, hidden_dim=hidden_size)
        self.classifier = BilinearMatrixAttention(
            hidden_size, hidden_size, use_input_biases=True, label_dim=self.dataset.num_labels
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
        all_hidden_states = []
        keys = (
            ("input_ids_1", "attention_mask_1", "token_type_ids_1"),
            ("input_ids_2", "attention_mask_2", "token_type_ids_2"),
        )
        for ids_key, mask_key, token_type_ids_key in keys:
            maybe_token_type_ids = (
                {"token_type_ids": batch[token_type_ids_key]} if token_type_ids_key in batch else {}
            )
            transformer_out = self.transformer(
                input_ids=batch[ids_key],
                attention_mask=batch[mask_key],
                **maybe_token_type_ids,
                output_hidden_states=self.hparams.scalar_mix,
            )
            if self.hparams.scalar_mix:
                hidden_state = self.scalar_mix(transformer_out.hidden_states, mask=batch[mask_key])
            else:
                hidden_state = transformer_out.last_hidden_state

            if self.hparams.mlp_layers > 0:
                hidden_state = self.mlp(hidden_state)

            all_hidden_states.append(hidden_state)
        hidden_state_1, hidden_state_2 = all_hidden_states

        # (bsz, hidden)
        pooled_1 = self.pooler(hidden_state_1, batch["attention_mask_1"])
        # (bsz, hidden)
        pooled_2 = self.pooler(hidden_state_2, batch["attention_mask_2"])

        logits = (
            self.classifier(pooled_1.unsqueeze(1), pooled_2.unsqueeze(1)).squeeze(-1).squeeze(-1)
        )  # (bsz, num_classes)
        return {"logits": logits}

    def setup_tokenizer(self) -> PreTrainedTokenizerBase:
        if self.hparams.tokenizer == "custom":
            return PlTokenizer()
        elif self.hparams.tokenizer == "pretrained":
            return AutoTokenizer.from_pretrained(self.hparams.model_name_or_path)
        elif self.hparams.tokenizer is None:
            return None
        else:
            assert False

    @staticmethod
    def add_args(parser):
        Model.add_args(parser)
        Pooler.add_args(parser)
        Transformer.add_args(parser)

        parser.add_argument("--mlp_layers", default=0, type=int)
        parser.add_argument("--scalar_mix", action="store_true")
        parser.add_argument("--tokenizer", type=str, default=None, choices=["pretrained", "custom"])
