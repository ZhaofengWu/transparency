import argparse
import os
import sys

import numpy as np
import torch
from tqdm import tqdm
from transformers import BertModel, GPT2Model, AutoTokenizer

from paoding import Transformer
from paoding.data.collator import collate_fn
from paoding.utils import add_parent_dir_to_path

add_parent_dir_to_path(__file__)

from data.text_pair_dataset import TextPairDataset


def load_examples(dataset):
    examples = []
    for split in ["train", "dev"] + dataset.test_splits:
        examples.extend(dataset.dataset_dict[split])
    return examples


def encode(model, input_ids, attention_mask):
    hidden_states = model(
        input_ids=input_ids.cuda(), attention_mask=attention_mask.cuda()
    ).last_hidden_state
    if isinstance(model.model, BertModel):
        pooled = hidden_states[:, 0]
    elif isinstance(model.model, GPT2Model):
        pooled = hidden_states[torch.arange(attention_mask.shape[0]), attention_mask.sum(-1) - 1]
    else:
        assert False
    return pooled.cpu().numpy()


def compute_embeddings(hparams, examples, batch_size):
    model = Transformer(hparams, "base", trainable=False).eval().cuda()
    with torch.no_grad():
        # For our synthetic dataset, example lengths are similar, so no need for sorting
        for i in tqdm(range(0, len(examples), batch_size), desc="Embedding"):
            curr_examples = [examples[j] for j in range(i, min(i + batch_size, len(examples)))]
            batch = collate_fn(
                curr_examples,
                "label",
                {
                    "input_ids_1": 0,
                    "attention_mask_1": False,
                    "input_ids_2": 0,
                    "attention_mask_2": False,
                },
                "right",
                "classification",
            )
            encoded_1 = encode(model, batch["input_ids_1"], batch["attention_mask_1"])
            encoded_2 = encode(model, batch["input_ids_2"], batch["attention_mask_2"])
            assert len(curr_examples) == len(encoded_1) == len(encoded_2)
            for example, e1, e2 in zip(curr_examples, encoded_1, encoded_2):
                example["embedding_1"] = e1
                example["embedding_2"] = e2


def cosine_similarity(X, Y):
    """
    X: (N, D) & Y: (N, D) -> (N,)
    """
    return np.einsum("nd,nd->n", X, Y) / (np.linalg.norm(X, axis=-1) * np.linalg.norm(Y, axis=-1))


def main(data_dir, transformer_model, output_file, batch_size=256):
    assert not os.path.exists(output_file)

    batch_size = int(batch_size)
    hparams = argparse.Namespace(
        data_dir=data_dir,
        data_type="nl",
        tokenizer="pretrained",
        max_length=1e6,
        model_name_or_path=transformer_model,
        seed=100,
        random_init_transformer=False,
        random_init_transformer_non_embeddings=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(transformer_model)
    dataset = TextPairDataset(hparams, tokenizer)

    examples = load_examples(dataset)
    compute_embeddings(hparams, examples, batch_size)

    embeddings_1 = np.stack([e["embedding_1"] for e in examples], axis=0)
    embeddings_2 = np.stack([e["embedding_2"] for e in examples], axis=0)

    print("Computing cos sim")
    sims = cosine_similarity(embeddings_1, embeddings_2)
    print("Done computing cos sim")

    with open(output_file, "w") as f:
        for s in sims:
            f.write(f"{str(s)}\n")


if __name__ == "__main__":
    main(*sys.argv[1:])  # pylint: disable=no-value-for-parameter
