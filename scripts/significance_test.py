import argparse
import random
import sys

import numpy as np
from tqdm import tqdm

from paoding.utils import add_parent_dir_to_path

add_parent_dir_to_path(__file__)

from data.text_pair_dataset import TextPairDataset

random.seed(100)


def avg(l):
    return sum(l) / len(l)


def flatten(l):
    return [e for subl in l for e in subl]


def bootstrap_by_verb(t_examples, o_examples, t_sims, o_sims, two_sided=True):
    orig_diff = np.mean(t_sims) - np.mean(o_sims)
    if two_sided:
        orig_diff = abs(orig_diff)

    t_ess = list(zip(t_examples, t_sims))
    o_ess = list(zip(o_examples, o_sims))
    t_ess_by_verb = {}
    o_ess_by_verb = {}
    for ess, ess_by_verb in ((t_ess, t_ess_by_verb), (o_ess, o_ess_by_verb)):
        for e, s in ess:
            verb = e["verb"]
            if verb not in ess_by_verb:
                ess_by_verb[verb] = []
            ess_by_verb[verb].append((e, s))

    diffs = []
    counter = 0
    for _ in range(1000):
        new_t_ess = flatten(random.choices(list(t_ess_by_verb.values()), k=len(t_ess_by_verb)))
        new_t_sims = np.array([s for e, s in new_t_ess])
        new_o_ess = flatten(random.choices(list(o_ess_by_verb.values()), k=len(o_ess_by_verb)))
        new_o_sims = np.array([s for e, s in new_o_ess])
        diff = np.mean(new_t_sims) - np.mean(new_o_sims)
        if two_sided:
            diff = abs(diff)
        diffs.append(diff)
        if diff > orig_diff:
            counter += 1
    print(counter, counter / 1000)


def permutation_by_verb(t_examples, o_examples, t_sims, o_sims, two_sided=True):
    orig_diff = np.mean(t_sims) - np.mean(o_sims)
    if two_sided:
        orig_diff = abs(orig_diff)

    t_ess = list(zip(t_examples, t_sims))
    o_ess = list(zip(o_examples, o_sims))
    sims_by_verb = {}
    for ess in (t_ess, o_ess):
        for e, sim in ess:
            verb = e["verb"]
            if verb not in sims_by_verb:
                sims_by_verb[verb] = []
            sims_by_verb[verb].append(sim)

    n = len(sims_by_verb)  # n is 12, so we can afford to do exact permutation
    counter = 0
    for decimal in tqdm(range(1, 2 ** n - 1)):  # 1, -1 s.t. we don't have empty slices
        bitstring = format(decimal, "b").zfill(n)
        assert len(bitstring) == n
        new_t_sims = []
        new_o_sims = []
        for bit, (verb, sims) in list(zip(bitstring, sims_by_verb.items())):
            (new_t_sims if bit == "1" else new_o_sims).extend(sims)
        diff = np.mean(new_t_sims) - np.mean(new_o_sims)
        if two_sided:
            diff = abs(diff)
        if diff >= orig_diff:
            counter += 1

    print(orig_diff, counter, counter / (2 ** n - 2))


def load_examples(orig_data_dir):
    hparams = argparse.Namespace(
        data_dir=orig_data_dir,
        data_type="nl",
        tokenizer="pretrained",
        max_length=1e6,
        seed=100,
        random_init_transformer=False,
        model_name_or_path="",
    )
    dataset = TextPairDataset(hparams, None, preprocess_and_save=False)
    examples = {
        k: list(v) for k, v in dataset.dataset_dict.items()
    }  # Dataset objects have weird behaviors when editting fields
    assert "test" not in examples
    examples["test"] = [e for split in dataset.test_splits for e in examples[split]]
    for split in dataset.test_splits:
        del examples[split]
    return examples["train"] + examples["dev"] + examples["test"]


def simplify_examples(
    examples,
    keep_keys={
        "ent_type",
        "fact_uuid",
        "label",
        "mod",
        "person",
        "second_text",
        "template_idx",
        "template_type",
        "text",
        "verb",
    },
):
    for e in examples:
        keys = e.keys() - keep_keys
        for k in keys:
            del e[k]


def main(orig_data_dir, similarity_file):
    examples = load_examples(orig_data_dir)
    sims = []
    with open(similarity_file) as f:
        for line in f:
            sims.append(float(line.strip("\n")))
    assert len(examples) == len(sims)

    examples, sims = zip(
        *[(e, s) for e, s in zip(examples, sims) if e["template_type"] == "attitude_embedded"]
    )
    print(len(examples))

    t_examples, t_sims = zip(
        *[(example, sim) for example, sim in zip(examples, sims) if example["label"] == 1]
    )
    o_examples, o_sims = zip(
        *[(example, sim) for example, sim in zip(examples, sims) if example["label"] == 0]
    )

    for examples in (t_examples, o_examples):
        simplify_examples(examples)

    print("avg t sim", avg(t_sims))
    print("avg o sim", avg(o_sims))
    print()

    print("bootstrap")
    bootstrap_by_verb(t_examples, o_examples, t_sims, o_sims)
    print("permutation")
    permutation_by_verb(t_examples, o_examples, t_sims, o_sims)


if __name__ == "__main__":
    main(*sys.argv[1:])  # pylint: disable=no-value-for-parameter
