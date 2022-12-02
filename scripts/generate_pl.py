"""
Generates the propositional logic corpus in the format of
((!(True|(False&False)))|....)=True
Generates four splits: pretrain, train, dev, test.
To generate the non-transparent version, set inplace_not=True.
In our experiments, transparent and non-transparent data have different EXPR_EXPANSION_PROB.
See our paper for more details.
"""

import random
import sys

from tqdm import tqdm

SYMBOLS = [True, False]
BINARY_OPS = ["&", "|"]
UNARY_OP = "!"
# The probability of EXPR := EXPR OP EXPR
EXPR_EXPANSION_PROB = 0.12
# The probability of EXPR := ! EXPR
UNARY_OP_PROB = (1 - EXPR_EXPANSION_PROB) / 2
MAX_LENGTH = 250
MAX_DEPTH = MAX_LENGTH

N_PRETRAIN = 51_200_000
N_TRAIN = 1_000_000
N_DEV = 10_000
N_TEST = 10_000

random.seed(77)


class Node:
    def __init__(self):
        self.left = self.right = self.repr = None
        self.value = 0


def sample(inplace_not=False, is_root=True, depth=1, flip_t=False, flip_f=False):
    curr = Node()
    expr_expansion_prob = EXPR_EXPANSION_PROB
    unary_op_prob_cum = expr_expansion_prob + UNARY_OP_PROB
    if is_root:  # we don't allow only one symbol, so reassign prob. mass
        expr_expansion_prob = expr_expansion_prob / unary_op_prob_cum
        unary_op_prob_cum = 1
    if depth == MAX_DEPTH:
        expr_expansion_prob = unary_op_prob_cum = -1
    rand = random.random()
    if rand < expr_expansion_prob:
        curr.left = sample(
            inplace_not=inplace_not, is_root=False, depth=depth + 1, flip_t=flip_t, flip_f=flip_f
        )
        left_binds = left_flip_t = left_flip_f = False
        if inplace_not:
            # These are not subject to higher flip_{t,f}, b/c it's the literal that triggers the flilp
            if curr.left.repr == "(!True)":
                left_binds = left_flip_t = True
            elif curr.left.repr == "(!False)":
                left_binds = left_flip_f = True

        curr.right = sample(
            inplace_not=inplace_not,
            is_root=False,
            depth=depth + 1,
            flip_t=flip_t or left_flip_t,
            flip_f=flip_f or left_flip_f,
        )
        if inplace_not:
            right_binds = right_flip_t = right_flip_f = False
            if curr.right.repr == "(!True)":
                right_binds = right_flip_t = True
            elif curr.right.repr == "(!False)":
                right_binds = right_flip_f = True

            if right_binds:
                if not left_binds:
                    # There are more clever ways than having to resample, but it'd require a 2-level
                    # lookahead. Trading efficiency for simplicity (and being less error-prone).
                    # We also need to make sure that the new left doesn't bind. We could do this via
                    # manipulating the probability mass, but since ! and T/F are sampled in two
                    # different levels, this can also be complicated and error prone.
                    while True:
                        curr.left = sample(
                            inplace_not=inplace_not,
                            is_root=False,
                            depth=depth + 1,
                            flip_t=flip_t or right_flip_t,
                            flip_f=flip_f or right_flip_f,
                        )
                        if curr.left.repr not in {"(!True)", "(!False)"}:
                            break
                else:
                    # We can't resample left, because the right subtree is dependent on left binding
                    # Luckily if left binds, there are only two possibilities, so we can enumerate.
                    if curr.left.repr == "(!True)" and right_flip_t:
                        curr.left.value = True
                    elif curr.left.repr == "(!False)" and right_flip_f:
                        curr.left.value = False

        op = random.choice(BINARY_OPS)
        left_value = curr.left.value
        right_value = curr.right.value

        curr.repr = f"({curr.left.repr}{op}{curr.right.repr})"
        if op == "&":
            curr.value = left_value and right_value
        elif op == "|":
            curr.value = left_value or right_value
        else:
            assert False
    elif rand < unary_op_prob_cum:
        # arbitrarily use the left child only for the unary op
        curr.left = sample(
            inplace_not=inplace_not, is_root=False, depth=depth + 1, flip_t=flip_t, flip_f=flip_f
        )
        curr.repr = f"(!{curr.left.repr})"
        curr.value = not curr.left.value
    else:
        sym = random.choice(SYMBOLS)
        curr.repr = str(sym)
        curr.value = not sym if ((sym and flip_t) or (not sym and flip_f)) else sym
    return curr


def sample_batch(target_size, inplace_not=False):
    assert target_size % 2 == 0
    samples = set()
    num_true = num_false = 0
    with tqdm(total=target_size) as progress:
        while len(samples) < target_size:
            sampled = sample(inplace_not=inplace_not)
            valued_repr = f"{sampled.repr}={sampled.value}"
            if (num_true >= target_size // 2 and sampled.value) or (
                num_false >= target_size // 2 and not sampled.value
            ):  # if we already have enough for one class, skip
                continue
            if valued_repr in samples:
                continue
            sample_len = (
                sum(sampled.repr.count(str(s)) for s in SYMBOLS + BINARY_OPS + [UNARY_OP, "(", ")"])
                + 2  # +2 for "=VALUE"
            )
            if sample_len > MAX_LENGTH:
                continue
            samples.add(valued_repr)
            if sampled.value:
                num_true += 1
            else:
                num_false += 1
            progress.update(1)

    assert len(samples) == target_size
    return samples


def write_output(samples, output_file):
    with open(output_file, "w") as f:
        for sample in samples:
            f.write(sample + "\n")


def main(output_file_prefix, inplace_not=False):
    splits = [
        ("pretrain.raw", N_PRETRAIN),
        ("train.raw", N_TRAIN),
        ("dev.raw", N_DEV),
        ("test.raw", N_TEST),
    ]
    samples = sample_batch(sum(split[1] for split in splits), inplace_not=inplace_not)

    # list of completed lists
    true = []
    false = []
    # current working list
    curr_true = []
    curr_false = []
    # current working index
    curr_true_split_idx = 0
    curr_false_split_idx = 0
    for sample in samples:
        value = sample.split("=")[1]
        assert value in ("True", "False")
        t = value == "True"
        (curr_true if t else curr_false).append(sample)
        if (
            len(curr_true if t else curr_false)
            == splits[curr_true_split_idx if t else curr_false_split_idx][1] // 2
        ):
            (true if t else false).append(curr_true if t else curr_false)
            if t:
                curr_true = []
                curr_true_split_idx += 1
            else:
                curr_false = []
                curr_false_split_idx += 1

    assert len(curr_true) == len(curr_false) == 0

    for i, split in enumerate(splits):
        split_samples = true[i] + false[i]
        random.shuffle(split_samples)
        write_output(split_samples, f"{output_file_prefix}.{split[0]}")


if __name__ == "__main__":
    main(*sys.argv[1:])  # pylint: disable=no-value-for-parameter,too-many-function-args
