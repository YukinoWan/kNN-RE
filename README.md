The repository contains the code of the paper "[Rescue Implicit and Long-tail Cases: Nearest Neighbor Relation Extraction.](https://aclanthology.org/2022.emnlp-main.113/)"

This respository is originally forked from [PURE](https://github.com/princeton-nlp/PURE)
## Setup

### Install dependencies
Please install all the dependency packages using the following command:
```
pip install -r requirements.txt
```


## Quick Start

### Input data format for the relation model
The input data format of the relation model is almost the same as that of the entity model, except that there is one more filed `."predicted_ner"` to store the predictions of the entity model.
```bash
{
  "doc_key": "CNN_ENG_20030306_083604.6",
  "sentences": [...],
  "ner": [...],
  "relations": [...],
  "predicted_ner": [
    [...],
    [...],
    [[26, 26, "LOC"], [14, 15, "PER"], ...],
    ...
  ]
}
```

### Train the relation model:
You can use `bash quick_run_i2b2.sh` to train a relation model for `i2b2` or other datasets correspondingly. A trianing command template is as follow:
```bash
python run_relation.py \
  --task {ace05 | tacred | scierc | i2b2 | wiki80} \
  --do_train --train_file {path to the training json file of the dataset} \
  --do_eval --eval_test --eval_with_gold \
  --model {bert-base-uncased | bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12 | allenai/scibert_scivocab_uncased} \
  --do_lower_case \
  --train_batch_size 32 \
  --eval_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 10 \
  --context_window 0 \
  --max_seq_length 128 \
  --entity_output_dir {path to output files of the entity model} \
  --output_dir {directory of output files} \
  --seed 0
```
Aruguments:
* `--eval_with_gold`: whether evaluate the model with the gold entities provided.
* `--entity_output_dir`: the output directory of the entity model. The prediction files (`ent_pred_dev.json` or `ent_pred_test.json`) of the entity model should be in this directory.

The prediction results will be stored in the file `predictions.json` in the folder `output_dir`, and the format will be almost the same with the output file from the entity model, except that there is one more field `"predicted_relations"` for each document.

You can run the evaluation script to output the end-to-end performance of the predictions.
```bash
python run_eval.py --prediction_file {path to output_dir}/predictions.json
```


### Memory construction:
You can use `bash save_datastore.sh` to construct memory for train, DS, dev, test. The stored memory will be used in kNN-RE. A save command template is as follow:
```bash
python knn_test.py \
  --task {ace05 | tacred | scierc | i2b2 | wiki80} \
  --do_eval --eval_test --eval_with_gold \
  --model {bert-base-uncased | bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12 | allenai/scibert_scivocab_uncased} \
  --do_lower_case \
  --train_batch_size 32 \
  --eval_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 10 \
  --context_window 0 \
  --max_seq_length 128 \
  --entity_output_dir {path to output files of the entity model} \
  --output_dir {directory of output files} \
  --seed 0 \
  --test_type {train | ds | dev | test}
```
Note that the model only evaluates and saves file `ent_pred_test.json`, thus you should first replace this file with {train | ds | dev | test} file you want to construct memory respectively. 
For example, to save train memory and test memory on i2b2, you can:
(1) save train memory:
```
cd dataset/i2b2
cp train.json ent_pred_test.json

cd ...
```
Then `bash save_datastore` with:
```
--task i2b2 \
--model bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12 \
--test_type train
```

(2) save test memory: 
```
cd dataset/i2b2
cp test.json ent_pred_test.json

cd ...
```
Then `bash save_datastore` with:
```
--task i2b2 \
--model bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12 \
--test_type test
```

### Memory construction
With the constructed memory for retrieval and test, you can run kNN-RE by `python knn_faiss.py {task} {train | ds } {dev | test}`, as the example above, with the train memory and test memory on i2b2, you can run `python knn_faiss.py i2b2 train test` to evaluate kNN-RE results.

