"""
Generates sentence pairs that demonstrate referential opacity, using the output of
`generate_facts.py`.
"""

from itertools import product
import json
import random
import sys

from tqdm import tqdm

random.seed(77)

# fmt: off

PERSONS = ["He", "She"]

VERBS = [
    {"form": "wants", "complement": "infinitive", "opacity": True},
    {"form": "intends", "complement": "infinitive", "opacity": True},
    {"form": "preferred", "complement": "infinitive", "opacity": True},
    {"form": "suggested", "complement": "infinitive", "opacity": True},
    {"form": "begs", "complement": "infinitive", "opacity": True},
    {"form": "hopes", "complement": "infinitive", "opacity": True},
    {"form": "managed", "complement": "infinitive", "opacity": False},
    {"form": "failed", "complement": "infinitive", "opacity": False},
    {"form": "starts", "complement": "infinitive", "opacity": False},
    {"form": "begins", "complement": "infinitive", "opacity": False},
    {"form": "ceases", "complement": "infinitive", "opacity": False},
    {"form": "stops", "complement": "infinitive", "opacity": False},
]

MODIFIERS = {
    "place": ["beautiful", "depressing"],
    "language": ["beautiful", "unattractive"],
    "corporation": ["well-known", "unknown"],
    "religion": ["well-known", "unknown"],
}

CLAUSE_TEMPLATES = [
    {"template": "[PERSON] [VERB] that [ENT] is [MOD]."},
]

INFINITIVE_TEMPLATES = [
    {"template": "[PERSON] [VERB] to learn about [ENT].", "blacklist_ent_types": {"language"}},
    {"template": "[PERSON] [VERB] to go to [ENT].", "whitelist_ent_types": {"place"}},
    {"template": "[PERSON] [VERB] to speak [ENT].", "whitelist_ent_types": {"language"}},
    {"template": "[PERSON] [VERB] to work for [ENT].", "whitelist_ent_types": {"corporation"}},
    {"template": "[PERSON] [VERB] to believe in [ENT].", "whitelist_ent_types": {"religion"}},
]

EMBEDDED_TEMPLATES = [t | {"required_complement": "clause" if t in CLAUSE_TEMPLATES else "infinitive"} for t in CLAUSE_TEMPLATES + INFINITIVE_TEMPLATES]

NON_EMBEDDED_TEMPLATES = [
    {"template": "[PERSON] is from [ENT].", "whitelist_ent_types": {"place"}, "opacity": False},
    {"template": "[PERSON] speaks [ENT].", "whitelist_ent_types": {"language"}, "opacity": False},
    {"template": "[PERSON] works for [ENT].", "whitelist_ent_types": {"corporation"}, "opacity": False},
    {"template": "[PERSON] dislikes [ENT].", "opacity": True},
]

# fmt: on


def fill_sentence(template, person, ent="", name="", verb="", mod=""):
    """
    [ENT] can be either a definite description or a name. [NAME] can only be a name.
    """
    return (
        template.replace("[PERSON]", person)
        .replace("[ENT]", ent)
        .replace("[NAME]", name)
        .replace("[VERB]", verb)
        .replace("[MOD]", mod)
    )


def fill_and_write(
    out, template, template_type, template_idx, person, fact, opacity, verb="", mod=""
):
    sentence_a = fill_sentence(
        template, person, ent=fact["desc"], name=fact["name"], verb=verb, mod=mod
    )
    sentence_b = fill_sentence(
        template, person, ent=fact["name"], name=fact["name"], verb=verb, mod=mod
    )
    pair_obj = {
        "fact_uuid": fact["uuid"],
        "fact_name": fact["name"],
        "fact_desc": fact["desc"],
        "text": sentence_a,
        "second_text": sentence_b,
        "label": 0 if opacity else 1,  # 1 iff paraphrase
        "template_type": template_type,
        "template_idx": template_idx,
        "person": person,
        "verb": verb,
        "mod": mod,
        "ent_type": fact["ent_type"],
    }
    out.write(json.dumps(pair_obj) + "\n")
    return pair_obj


def template_admits_fact(template, fact):
    if fact["ent_type"] in template.get("blacklist_ent_types", set()):
        return False
    if (
        "whitelist_ent_types" in template
        and fact["ent_type"] not in template["whitelist_ent_types"]
    ):
        return False
    return True


def main(facts_file, output_file):
    facts = []
    with open(facts_file) as f:
        for line in f:
            facts.append(json.loads(line))

    pairs = []
    with open(output_file, "w") as f:
        # These templates don't have a verb slot to be filled
        for i, template in enumerate(tqdm(NON_EMBEDDED_TEMPLATES)):
            for fact in facts:
                if fact["ent_type"] == "religion":
                    # for balance
                    continue
                if not template_admits_fact(template, fact):
                    continue
                for person in PERSONS:
                    pairs.append(
                        fill_and_write(
                            f,
                            template["template"],
                            "attitude_nonembedded",
                            i,
                            person,
                            fact,
                            template["opacity"],
                        )
                    )

        for verb in tqdm(VERBS):
            for i, template in enumerate(EMBEDDED_TEMPLATES):
                if (
                    template["required_complement"] != verb["complement"]
                    and verb["complement"] != "both"
                ):
                    continue
                for fact in facts:
                    if not template_admits_fact(template, fact):
                        continue
                    for person in PERSONS:
                        if "[MOD]" not in template["template"]:
                            pairs.append(
                                fill_and_write(
                                    f,
                                    template["template"],
                                    "attitude_embedded",
                                    i,
                                    person,
                                    fact,
                                    verb["opacity"],
                                    verb=verb["form"],
                                )
                            )
                        else:
                            for mod in MODIFIERS[fact["ent_type"]]:
                                pairs.append(
                                    fill_and_write(
                                        f,
                                        template["template"],
                                        "attitude_embedded",
                                        i,
                                        person,
                                        fact,
                                        verb["opacity"],
                                        verb=verb["form"],
                                        mod=mod,
                                    )
                                )

        # --- Coordinated pairs ---

        uuids = {pair["fact_uuid"] for pair in pairs}
        # Esmitated ratio from unsubsampled trial such as there are approx. as many coordinated
        # pairs as non-coordinated pairs
        subsample_ratio = 11.57222736
        subsampled_uuids = set(random.sample(uuids, int(len(uuids) / subsample_ratio)))

        random.shuffle(pairs)
        for l_idx, pair_l in enumerate(tqdm(pairs)):
            for r_idx, pair_r in enumerate(pairs):
                if r_idx >= l_idx:
                    break
                if pair_l["person"] == pair_r["person"]:
                    # So the conjuncts are completely independent, making the data cleaner
                    continue
                if (
                    pair_l["fact_uuid"] != pair_r["fact_uuid"]
                    or pair_l["fact_uuid"] not in subsampled_uuids
                ):
                    continue
                assert all(pair_l[k] == pair_r[k] for k in ("fact_name", "fact_desc", "ent_type"))
                assert pair_l["text"][-1] == pair_l["second_text"][-1] == "."
                pos_pair = neg_pair = None
                keyss = list(product(("text", "second_text"), repeat=4))
                random.shuffle(keyss)
                for keys in keyss:
                    if keys[0] == keys[2] and keys[1] == keys[3]:
                        continue  # otherwise the two new sentences are the same
                    combine = lambda l, r: l[:-1] + " and " + r[0].lower() + r[1:]
                    coordinated_pair = (
                        {k + "_l": v for k, v in pair_l.items()}
                        | {k + "_r": v for k, v in pair_r.items()}
                        | {
                            "text": combine(pair_l[keys[0]], pair_r[keys[1]]),
                            "second_text": combine(pair_l[keys[2]], pair_r[keys[3]]),
                            "label": 1
                            if (pair_l["label"] == 1 or keys[0] == keys[2])
                            and (pair_r["label"] == 1 or keys[1] == keys[3])
                            else 0,
                            "template_type": "coordinate_"
                            + "".join({"text": "1", "second_text": "2"}[k] for k in keys),
                        }
                    )
                    if coordinated_pair["label"] == 1:
                        if pos_pair is not None:
                            continue
                        pos_pair = coordinated_pair
                    if coordinated_pair["label"] == 0:
                        if neg_pair is not None:
                            continue
                        neg_pair = coordinated_pair
                if pos_pair is None or neg_pair is None:  # ensure balance
                    continue
                f.write(json.dumps(pos_pair) + "\n")
                f.write(json.dumps(neg_pair) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:])  # pylint: disable=no-value-for-parameter
