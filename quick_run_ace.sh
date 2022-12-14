python run_relation.py \
      --task ace05  \
        --do_train --train_file "./dataset/ace05data/train.json" \
          --do_eval --eval_test --eval_with_gold \
            --model bert-base-uncased  \
              --do_lower_case \
                --train_batch_size 32 \
                  --eval_batch_size 32 \
                    --learning_rate 2e-5 \
                      --num_train_epochs 10 \
                        --context_window 0 \
                          --max_seq_length 128 \
                            --entity_output_dir "./dataset/ace05data" \
                              --output_dir ace05_relation_bertbase_seed \
                              --seed 3
# Output end-to-end evaluation results
python run_eval.py --prediction_file ace05_relation_bertbase_seed/predictions.json

