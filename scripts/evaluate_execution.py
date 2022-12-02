"""
Evaluates zero-shot execution accuracy of a pretrained LM.
"""

import sys

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoConfig, GPT2Config, RobertaConfig, GPT2LMHeadModel, RobertaForMaskedLM

from paoding.data.collator import collate_fn
from paoding.utils import add_parent_dir_to_path

add_parent_dir_to_path(__file__)

from data.pl_tokenizer import PlTokenizer

MODEL_CLASSES = {GPT2Config: GPT2LMHeadModel, RobertaConfig: RobertaForMaskedLM}


def main(ckpt_path, raw_test_file, batch_size):
    batch_size = int(batch_size)

    config = AutoConfig.from_pretrained(ckpt_path)
    model_class = MODEL_CLASSES[type(config)]
    model = model_class.from_pretrained(ckpt_path, config=config).cuda().eval()
    with open(raw_test_file) as f:
        lines = [line.strip() for line in f]

    templates = [
        ["(", "True", "&", "[MASK]", ")"],
        ["(", "False", "|", "[MASK]", ")"],
        ["(", "[MASK]", "&", "True", ")"],
        ["(", "[MASK]", "|", "False", ")"],
        ["(", "!", "[MASK]", ")"],
    ]
    # Whether the prediction should be flipped  for each template
    inverts = [False, False, False, False, True]

    tokenizer = PlTokenizer()
    value_length = 1
    accs = []
    for template, invert in zip(templates, inverts):
        mask_idx = template.index("[MASK]")

        n_correct = 0
        for i in tqdm(range(0, len(lines), batch_size)):
            # The following is a bit hacky and assumes knowledge of how the (custom) tokenizers work
            # internally
            batch = [tokenizer(line) for line in lines[i : i + batch_size]]
            actual_batch_size = len(batch)  # could be different from batch_size for the last batch
            bsz_arange = torch.arange(actual_batch_size)
            labels = [e["input_ids"][-value_length:] for e in batch]
            batch = [
                {
                    "input_ids": e["input_ids"][:-value_length],
                    "attention_mask": e["attention_mask"][:-value_length],
                }
                for e in batch
            ]

            if isinstance(config, GPT2Config):
                candidates = []
                for v in ("True", "False"):
                    template_copy = list(template)
                    template_copy[mask_idx] = v
                    candidates.append(tokenizer.convert_tokens_to_ids(template_copy))

                cand_len = len(candidates[0])
                assert all(len(cand) == cand_len for cand in candidates)

                new_batch = []
                for e in batch:
                    for cand in candidates:
                        new_batch.append(
                            {
                                "input_ids": e["input_ids"] + cand,
                                "attention_mask": e["attention_mask"] + [1] * cand_len,
                            }
                        )
                batch = new_batch
                candidates = torch.tensor(candidates).cuda()
            else:  # roberta
                batch = [
                    {
                        "input_ids": e["input_ids"]
                        + [
                            tokenizer.convert_tokens_to_ids(x)
                            if x != "[MASK]"
                            else tokenizer.mask_token_id
                            for x in template
                        ],
                        "attention_mask": e["attention_mask"] + [1] * len(template),
                    }
                    for e in batch
                ]
                mask_offset = mask_idx - len(template)

            batch = collate_fn(
                batch, None, {"input_ids": 0, "attention_mask": False}, "right", "classification"
            )
            input_ids, attention_mask = batch["input_ids"], batch["attention_mask"]
            input_ids, attention_mask = input_ids.cuda(), attention_mask.cuda()
            labels = torch.tensor(labels).cuda()

            with torch.no_grad():
                output = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = output.logits  # (bsz, seq_len, vocab_size)

            if isinstance(config, GPT2Config):
                assert len(logits) == len(attention_mask)
                rolled_logits = torch.stack(
                    [
                        l.roll(-m.sum().item() + cand_len + 1, 0)
                        for l, m in zip(logits, attention_mask)
                    ],
                    0,
                )  # l: (seq_len, vocab_size), m: (seq_len,)
                segmented_logits = rolled_logits[:, :cand_len, :].reshape(
                    actual_batch_size, len(candidates), cand_len, rolled_logits.shape[-1]
                )
                pred_logits = (
                    segmented_logits.gather(
                        -1, candidates.unsqueeze(0).expand(actual_batch_size, -1, -1).unsqueeze(-1)
                    )
                    .squeeze(-1)
                    .sum(-1)
                )  # (bsz, num_cand)
                labels = labels.squeeze(1) == tokenizer.convert_tokens_to_ids("False")
                pred = pred_logits.argmax(1)
                if not invert:
                    n_correct += (pred == labels).sum().item()
                else:
                    n_correct += (pred != labels).sum().item()
            else:  # roberta
                true_plus_false = sum(tokenizer.convert_tokens_to_ids(["True", "False"]))
                labels = labels.squeeze(-1)
                logits = logits[bsz_arange, attention_mask.sum(-1) + mask_offset, :]
                target = logits[bsz_arange, labels]
                non_target = logits[bsz_arange, true_plus_false - labels]
                if not invert:
                    n_correct += (target >= non_target).sum().item()
                else:
                    n_correct += (target < non_target).sum().item()

        accs.append(n_correct / len(lines))

    print(f"Accuracy: {np.mean(accs):.4f} ± {np.std(accs, ddof=1):.4f}")


if __name__ == "__main__":
    main(*sys.argv[1:])  # pylint: disable=no-value-for-parameter
