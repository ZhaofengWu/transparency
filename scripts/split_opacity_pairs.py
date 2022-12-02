"""
Splits the referential opacity sentence pairs from `generate_opacity_pairs.py` to train/dev/test
sets.
"""

import json
import random
import os
import sys

random.seed(1)


def split_list(l):
    cut_1 = len(l) // 10 * 8
    cut_2 = len(l) // 10 * 9
    return l[:cut_1], l[cut_1:cut_2], l[cut_2:]


def parse_bool(flag):
    if flag in {"True", "False"}:
        return flag == "True"
    else:
        assert isinstance(flag, bool)
        return flag


def main(opacity_pairs_file, output_dir, include_coordinate=True):
    include_coordinate = parse_bool(include_coordinate)

    pairs = []
    with open(opacity_pairs_file) as f:
        for line in f:
            pairs.append(json.loads(line))

    common_keys = None
    for pair in pairs:
        if not include_coordinate and pair["template_type"].startswith("coordinate"):
            continue
        if common_keys is None:
            common_keys = set(pair.keys())
        else:
            common_keys &= pair.keys()

    simple_pos_pairs = []
    simple_neg_pairs = []
    if include_coordinate:
        coordinate_pos_pairs = []
        coordinate_neg_pairs = []
    for pair in pairs:
        assert pair["label"] in (0, 1)
        if pair["label"] == 1:
            if not pair["template_type"].startswith("coordinate"):
                simple_pos_pairs.append(pair)
            elif include_coordinate:
                coordinate_pos_pairs.append(pair)
        else:
            if not pair["template_type"].startswith("coordinate"):
                simple_neg_pairs.append(pair)
            elif include_coordinate:
                coordinate_neg_pairs.append(pair)

    assert len(simple_pos_pairs) == len(simple_neg_pairs)
    if include_coordinate:
        assert len(coordinate_pos_pairs) == len(coordinate_neg_pairs)
    for pairs in (simple_pos_pairs, simple_neg_pairs) + (
        (coordinate_pos_pairs, coordinate_neg_pairs) if include_coordinate else tuple()
    ):
        random.shuffle(pairs)

    split_simple_pos_pairs = split_list(simple_pos_pairs)
    split_simple_neg_pairs = split_list(simple_neg_pairs)
    if include_coordinate:
        split_coordinate_pos_pairs = split_list(coordinate_pos_pairs)
        split_coordinate_neg_pairs = split_list(coordinate_neg_pairs)

    train_pairs = split_simple_pos_pairs[0] + split_simple_neg_pairs[0]
    dev_pairs = split_simple_pos_pairs[1] + split_simple_neg_pairs[1]
    test_pairs = {
        "test_simple_1": split_simple_pos_pairs[2],
        "test_simple_0": split_simple_neg_pairs[2],
    }
    if include_coordinate:
        train_pairs.extend(split_coordinate_pos_pairs[0] + split_coordinate_neg_pairs[0])
        dev_pairs.extend(split_coordinate_pos_pairs[1] + split_coordinate_neg_pairs[1])
        test_pairs |= {
            "test_coordinate_1": split_coordinate_pos_pairs[2],
            "test_coordinate_0": split_coordinate_neg_pairs[2],
        }

    random.shuffle(train_pairs)
    random.shuffle(dev_pairs)
    for pairs in test_pairs.values():
        random.shuffle(pairs)

    for split_name, split in ({"train": train_pairs, "dev": dev_pairs} | test_pairs).items():
        print(f"{split_name}: {len(split)}")
        with open(os.path.join(output_dir, f"nl.{split_name}"), "w") as f:
            for pair in split:
                pair = {k: v for k, v in pair.items() if k in common_keys}
                f.write(json.dumps(pair) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:])  # pylint: disable=no-value-for-parameter
