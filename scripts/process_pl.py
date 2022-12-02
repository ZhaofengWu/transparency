"""
Processes the propositional logic corpus from the format of
((!(True|(False&False)))|....)=True
to the format for equivalence checking
((!(True|(False&False)))|....)\t((True|(!(True&False)))|....)\t1
"""

import os
import sys
import random

from tqdm import tqdm

random.seed(77)


def main(input_file, output_file):
    assert not os.path.exists(output_file)

    lines = []
    with open(input_file) as f:
        for line in f:
            lines.append(line)

    target_size = len(lines)
    assert target_size % 2 == 0
    num_same = num_diff = 0
    with open(output_file, "w") as f:
        with tqdm(total=target_size) as progress:
            while num_same + num_diff < target_size:
                line1 = random.choice(lines)
                line2 = random.choice(lines)
                if line1 == line2:
                    continue
                expr1, label1 = line1.split("=")
                expr2, label2 = line2.split("=")
                if (num_same >= target_size // 2 and label1 == label2) or (
                    num_diff >= target_size // 2 and label1 != label2
                ):  # if we already have enough for one class, skip
                    continue
                f.write(f"{expr1}\t{expr2}\t{1 if label1 == label2 else 0}\n")
                if label1 == label2:
                    num_same += 1
                else:
                    num_diff += 1
                progress.update(1)


if __name__ == "__main__":
    main(*sys.argv[1:])  # pylint: disable=no-value-for-parameter
