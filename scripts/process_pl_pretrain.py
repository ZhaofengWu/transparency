"""
Processes the propositional logic corpus from the format of
((!(True|(False&False)))|....)=True
to the format for the pretraining setup
((!(True|(False&False)))|....)=((True|(!(True&False)))|....)

Also ensures grounding (see the paper for details).

To generate the non-grounded version, do not set clique_size.
To generate the +reflexivity, -symmetry version, set reflexivity_only=True.
To generate the -reflexivity, +symmetry version, set reflexivity=False.
"""

import sys
import random

from tqdm import tqdm

random.seed(77)


def parse_bool(flag):
    if flag in {"True", "False"}:
        return flag == "True"
    else:
        assert isinstance(flag, bool)
        return flag


def main(
    input_file,
    output_file,
    clique_size=None,
    target_size=None,
    reflexivity=True,
    reflexivity_only=False,
):
    reflexivity = parse_bool(reflexivity)
    reflexivity_only = parse_bool(reflexivity_only)
    if clique_size == "None":
        clique_size = None
    if clique_size is not None:
        clique_size = int(clique_size)

    lines = []
    with open(input_file) as f:
        for line in f:
            lines.append(line)

    if clique_size is not None:
        t_lines = [line for line in lines if line.strip().split("=")[1] == "True"]
        f_lines = [line for line in lines if line.strip().split("=")[1] == "False"]
        assert len(t_lines) == len(f_lines) == len(lines) // 2

    target_size = int(target_size) if target_size is not None else len(lines)
    sampled = 0
    with open(output_file, "w") as f:
        with tqdm(total=target_size) as progress:
            while sampled < target_size:
                if clique_size is not None:
                    clique_lines = random.sample(random.choice((t_lines, f_lines)), clique_size)
                    clique_lines = [line.strip() for line in clique_lines]
                    for i, line1 in enumerate(clique_lines):
                        for j, line2 in enumerate(clique_lines):
                            if not reflexivity and line1 == line2:
                                continue
                            if reflexivity_only and j > i:
                                continue
                            lhs, label1 = line1.split("=")
                            rhs, label2 = line2.split("=")
                            assert label1 == label2
                            f.write(f"{lhs}={rhs}\n")
                            sampled += 1
                            progress.update(1)
                    continue

                line1 = random.choice(lines).strip()
                line2 = random.choice(lines).strip()
                if line1 == line2:
                    continue
                lhs, label1 = line1.split("=")
                rhs, label2 = line2.split("=")
                if label1 != label2:
                    continue
                f.write(f"{lhs}={rhs}\n")
                sampled += 1
                progress.update(1)


if __name__ == "__main__":
    main(*sys.argv[1:])  # pylint: disable=no-value-for-parameter
