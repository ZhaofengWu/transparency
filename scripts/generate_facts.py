"""
Generate a list of facts from TREx that GPT-2 knows。
For example, the official language of China == Chinese.
"""

import json
import os
import re
import sys

from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer, BertTokenizer, BertForMaskedLM

TOKENIZERS = {"gpt2-xl": GPT2Tokenizer, "bert-large-cased": BertTokenizer}
MODELS = {"gpt2-xl": GPT2LMHeadModel, "bert-large-cased": BertForMaskedLM}


def init_lm(model_name):
    tokenizer = TOKENIZERS[model_name].from_pretrained(model_name)
    model = MODELS[model_name].from_pretrained(model_name).eval().cuda()
    return (tokenizer, model)


def run_lm(tokenizer, model, prefix, max_len) -> str:
    if isinstance(model, GPT2LMHeadModel):
        input_ids = tokenizer.encode(prefix, return_tensors="pt").cuda()
        output_ids = model.generate(
            input_ids,
            max_length=len(input_ids[0]) + max_len,
            do_sample=False,
            top_k=0,
            num_beams=5,
            early_stopping=True,
            pad_token_id=model.config.eos_token_id,
        )
        output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        # We want the prefix to be a strict, well, prefix, of the output, but the stupic GPT-2
        # tokenization doesn't guarantee this in this case. So we manually patch it
        prefix = re.sub(r"([a-z]) \.", r"\1.", prefix)

        assert output[: len(prefix)] == prefix
        return output[len(prefix) :].strip()
    elif isinstance(model, BertForMaskedLM):
        input_ids = tokenizer.encode(
            f"{prefix} {tokenizer.mask_token} .", return_tensors="pt"
        ).cuda()
        logits = model(input_ids).logits[0, -3]  # "xxx [MASK] . [SEP]", so -3
        return tokenizer.convert_ids_to_tokens([logits.argmax()])[0]
    else:
        raise NotImplementedError


def write_output(out, rel, obj, desc, name):
    out.write(
        json.dumps(
            {
                "uuid": obj["uuid"],
                "desc": desc,
                "name": name,
                "rel_id": rel["relation"],
                "relation": rel["label"],
                "rel_type": rel["type"],
                "ent_type": rel["obj_type"],
            }
        )
        + "\n"
    )


def main(model_name, lama_dir, output_file):
    tokenizer, model = init_lm(model_name)

    relations = {}
    with open(os.path.join(lama_dir, "filtered_relations.jsonl")) as f:
        for line in f:
            obj = json.loads(line)
            assert obj["relation"] not in relations
            relations[obj["relation"]] = obj

    agg_correct = agg_total = 0
    with open(output_file, "w") as out:
        for i, (rel_id, rel) in enumerate(relations.items()):
            print(rel_id, rel["label"], rel["type"])
            type_ = rel["type"]
            assert type_ in {"1-1", "N-1"}
            correct = total = 0

            file_name = os.path.join(lama_dir, "TREx", f"{rel_id}.jsonl")
            num_lines = sum(1 for line in open(file_name))
            with open(file_name) as f:
                for line in tqdm(f, desc=f"{i + 1} of {len(relations)}", total=num_lines):
                    obj = json.loads(line)
                    assert "[X]" in rel["template"]
                    desc = rel["template"].replace("[X]", obj["sub_label"])
                    target = obj["obj_label"]
                    target_len = len(tokenizer.encode(" " + target))

                    assert desc[0] == "t"
                    Desc = "T" + desc[1:]  # capitalize

                    prompt = f"{Desc} is known as"
                    pred = run_lm(tokenizer, model, prompt, target_len)

                    if pred == target or pred == target + ".":
                        write_output(out, rel, obj, desc, pred)
                        correct += 1
                        agg_correct += 1
                    total += 1
                    agg_total += 1
            out.flush()
            print(f"Accuracy: {correct / total}")
            print()

    print("=" * 30)
    print(f"Total accuracy: {agg_correct / agg_total}")


if __name__ == "__main__":
    main(*sys.argv[1:])  # pylint: disable=no-value-for-parameter
