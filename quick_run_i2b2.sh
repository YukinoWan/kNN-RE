python run_relation.py \
      --task i2b2  \
        --do_train --train_file "./dataset/i2b2/train.json" \
          --do_eval --eval_test --eval_with_gold \
            --model bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12  \
              --do_lower_case \
                --train_batch_size 32 \
                  --eval_batch_size 32 \
                    --learning_rate 2e-5 \
                      --num_train_epochs 10 \
                        --context_window 0 \
                          --max_seq_length 128 \
                            --entity_output_dir "./dataset/i2b2" \
                              --output_dir i2b2_relation_bertbase_seed \
                              --seed 1
# Output end-to-end evaluation results
python run_eval.py --prediction_file i2b2_relation_bertbase_seed/predictions.json

