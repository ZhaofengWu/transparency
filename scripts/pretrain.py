"""
Pretrains a language model on the pretraining corpus of propositional logic.

Referencing https://huggingface.co/blog/how-to-train and
https://colab.research.google.com/github/huggingface/blog/blob/master/notebooks/01_how_to_train.ipynb and
https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/language_modeling_from_scratch.ipynb and
https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_mlm.py
"""

import os
import sys
from typing import Any, Optional, Tuple

import datasets
from pytorch_lightning import seed_everything
from transformers import AutoConfig
from transformers import DataCollatorForLanguageModeling as HFDataCollatorForLanguageModeling
from transformers import RobertaForMaskedLM, GPT2LMHeadModel
from transformers import Trainer, TrainingArguments

from paoding.utils import add_parent_dir_to_path

add_parent_dir_to_path(__file__)

from data.pl_tokenizer import PlTokenizer
from data.text_pair_dataset import NUM_PRETRAINING_EMBEDDINGS

MAX_LEN = 512
EFFECTIVE_BATCH_SIZES = {"roberta-base": 8192, "gpt2": 8192}
MODEL_CLASSES = {"roberta-base": RobertaForMaskedLM, "gpt2": GPT2LMHeadModel}
MODEL_TYPES = {"roberta-base": "mlm", "gpt2": "clm"}
ADAM_EPSILONS = {"roberta-base": 1e-6, "gpt2": 1e-8}
ADAM_BETA2S = {"roberta-base": 0.98, "gpt2": 0.95}

seed_everything(100)


def main(
    data_path,
    model_name,
    batch_size,
    output_dir,
):
    assert model_name in {"roberta-base", "gpt2"}
    batch_size = int(batch_size)
    effective_batch_size = EFFECTIVE_BATCH_SIZES[model_name]
    assert effective_batch_size % batch_size == 0

    model_cls = MODEL_CLASSES[model_name]
    model_type = MODEL_TYPES[model_name]

    config = AutoConfig.from_pretrained(model_name)
    config.vocab_size = NUM_PRETRAINING_EMBEDDINGS
    model = model_cls(config=config)
    tokenizer = PlTokenizer()

    def tokenize(examples):
        examples["text"] = [
            line for line in examples["text"] if len(line) > 0 and not line.isspace()
        ]
        return tokenizer(examples["text"], truncation=True, max_length=MAX_LEN)

    dataset = datasets.load_dataset("text", data_files=data_path)["train"]
    dataset = dataset.map(
        tokenize,
        batched=True,
        num_proc=8,
        remove_columns=["text"],
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=(model_type == "mlm"), mlm_probability=0.15
    )

    gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    assert gpus >= 1
    training_args = TrainingArguments(
        output_dir=output_dir,
        group_by_length=True,
        dataloader_num_workers=2,
        num_train_epochs=1,  # should be exactly 100k steps
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=effective_batch_size // (gpus * batch_size),
        save_steps=5_000,
        warmup_steps=10_000,
        learning_rate=6e-4,
        weight_decay=0.1,
        adam_epsilon=ADAM_EPSILONS[model_name],
        adam_beta2=ADAM_BETA2S[model_name],
        fp16=True,
        prediction_loss_only=True,
        report_to="tensorboard",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    trainer.train()
    trainer.save_model(output_dir)


# fmt: off


class DataCollatorForLanguageModeling(HFDataCollatorForLanguageModeling):
    def torch_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        """
        The same as the super class, but when changing to a random token, we only want "real" tokens
        i.e. no special tokens or reserved tokens. The only change is marked with <CHANGE></CHANGE>
        """
        import torch

        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        # <CHANGE>
        real_token_indices = torch.LongTensor(self.tokenizer.real_token_indices())
        sampled_indices = torch.randint(len(real_token_indices), labels.shape, dtype=torch.long)
        random_words = real_token_indices[sampled_indices]
        # </CHANGE>
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

# fmt: on

if __name__ == "__main__":
    main(*sys.argv[1:])  # pylint: disable=no-value-for-parameter
