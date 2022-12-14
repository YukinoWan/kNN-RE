python run_relation.py \
      --task tacred  \
        --do_train --train_file "./dataset/tacred_pure/train.json" \
          --do_eval --eval_test --eval_with_gold \
            --model bert-base-uncased  \
              --do_lower_case \
                --train_batch_size 32 \
                  --eval_batch_size 32 \
                    --learning_rate 2e-5 \
                      --num_train_epochs 10 \
                        --context_window 0 \
                          --max_seq_length 128 \
                            --entity_output_dir "./dataset/tacred_pure" \
                              --output_dir tacred_relation_bertbase_seed0 \
                              --seed 0
# Output end-to-end evaluation results
python run_eval.py --prediction_file tacred_relation_bertbase_seed0/predictions.json

