# Transparency Helps Reveal When Language Models Learn Meaning

The official implementation for our paper (https://arxiv.org/abs/2210.07468):

```bibtex
@inproceedings{wu-etal-2022-continued,
    title = "Transparency Helps Reveal When Language Models Learn Meaning",
    author = "Zhaofeng Wu and William Merrill and Hao Peng and Iz Beltagy and Noah A. Smith",
    url = {https://arxiv.org/abs/2210.07468},
    publisher = {arXiv},
    year = {2022},
    doi = {10.48550/ARXIV.2210.07468},
}
```

## Environment

All experiments were performed with Python 3.9.7. The only dependency is the [PaoDing](https://github.com/ZhaofengWu/PaoDing) library, which is a (somewhat personal?) infrastructure library that I developed. Install it with `pip install PaoDing==0.1.1` (tag `0.1.0` in the repo has the exact same dependencies we used, which contains a direct dependency of a specifc commit of AllenNLP, which apparently PyPi doesn't like. Tag `0.1.1`, which is on PyPi, uses the closest AllenNLP release, which I think should be no different for our purposes).

## Propositional Logic Experiments

### Data

We release our generated datasets, both the transparent and the non-transparent versions, at https://huggingface.co/datasets/ZhaofengWu/transparency-data/tree/main/propositional_logic. Note that because the files are large, they may not load correctly on Windows.

#### Re-generate Data

You could also regenerate them with the following commands, though it may take a few days. See the file-top comments for more details, including options that generate variations of the datasets we explored.

```bash
data_dir=datasets/propositional_logic/transparent
mkdir -p ${data_dir}
# Generate the dataset in the format of (T&(F|T))=T
python scripts/generate_pl.py ${data_dir}/pl
# Processes the dataset into the format of (T&(F|T))=(F|(T&T)), ensuring grounding
python scripts/process_pl_pretrain.py ${data_dir}/pl.pretrain.raw ${data_dir}/pl.pretrain 2 819200000
for split in train dev test; do python scripts/process_pl.py ${data_dir}/pl.${split}.raw ${data_dir}/pl.${split}; done
```

### Pretraining

We release our pretrained models at https://huggingface.co/ZhaofengWu/transparency-models/tree/main/propositional_logic. You could also pretrain them with the following commands, assuming `${data_dir}` is set to be the same as above. Note that pretraining may take a few weeks.

```bash
pretrained_model_dir=output/pretrained_model
py scripts/pretrain.py ${data_dir}/pl.pretrain gpt2 64 ${pretrained_model_dir}  # or change to roberta-base
```

### Equivalence Probing

Assuming `${data_dir}` and `${pretrained_model_dir}` are set to be the same as above, run:

```bash
python train.py --data_dir ${data_dir} --data_type pl --model_name_or_path ${pretrained_model_dir}/checkpoint-100000 --tokenizer custom --pooling_mode {last or avg or attn_k, depending on the model type and ±attn, see our paper} {add --scalar_mix for MLM, see our paper} --batch_size 8 --lr 0.00001 --warmup_steps 1000 --output_dir ${output_dir} --epochs 3 --clip_norm 1.0
```

### Direct Evaluation

Assuming `${data_dir}` and `${pretrained_model_dir}` are set to be the same as above, run:

```bash
python scripts/evaluate_execution.py ${pretrained_model_dir} ${data_dir}/pl.test.raw 32
```

## Natural Language Experiments

### Data

We release our generated datasets, both with and without coordinated pairs and with GPT2-XL or BERT-large-cased, at https://huggingface.co/datasets/ZhaofengWu/transparency-data/tree/main/natural_language.

#### Re-generate Data

You could also regenerate them yourself, in which case you need to download the LAMA data following instructions in https://github.com/facebookresearch/LAMA. The script below assumes the lama directory to look like:

```
- TREx
  - P*.jsonl
- filtered_relations.jsonl
```

where `filtered_relations.jsonl` is curated by us and has content

```
{"relation": "P37", "template": "the official language of [X]", "label": "official language", "description": "language designated as official by this item", "type": "N-1", "obj_type": "language"}
{"relation": "P176", "template": "the manufacturer that produced [X]", "label": "manufacturer", "description": "manufacturer or producer of this product", "type": "N-1", "obj_type": "corporation"}
{"relation": "P140", "template": "the religion associated with [X]", "label": "religion", "description": "religion of a person, organization or religious building, or associated with this subject", "type": "N-1", "obj_type": "religion"}
{"relation": "P103", "template": "the native language of [X]", "label": "native language", "description": "language or languages a person has learned from early childhood", "type": "N-1", "obj_type": "language"}
{"relation": "P36", "template": "the capital of [X]", "label": "capital", "description": "primary city of a country, state or other type of administrative territorial entity", "type": "1-1", "obj_type": "place"}
```

Then run:

```bash
mkdir ${data_dir}
python scripts/generate_facts.py {gpt2-xl or bert-large-cased} ${lama_dir} ${facts_file}
python scripts/generate_opacity_pairs.py ${facts_file} ${opacity_pairs_file}
python scripts/split_opacity_pairs.py ${opacity_pairs_file} ${data_dir}
```

### Equivalence Probing

Use the following comands for training and evaluation:

```bash
python train.py --data_dir ${data_dir} --data_type nl --model_name_or_path {gpt2-xl or bert-large-cased} --tokenizer pretrained --pooling_mode {last or avg, depending on the model type, see our paper} {add --scalar_mix for MLM, see our paper} --batch_size 256 --lr 0.00001 --warmup_steps 1000 --output_dir ${output_dir} --epochs 1 --clip_norm 1.0
python evaluate.py --ckpt_path ${output_dir}/best.ckpt --splits test
```

### Sentence Pair Similarity

Dump the cosine similarity between each sentence pair with:

```bash
python scripts/write_cosine_similarity.py ${data_dir} {gpt2-xl or bert-large-cased} ${cos_sim_file}
```

Then run the following command to perform the significance tests:

```bash
python scripts/significance_test.py ${data_dir} ${cos_sim_file}
```
