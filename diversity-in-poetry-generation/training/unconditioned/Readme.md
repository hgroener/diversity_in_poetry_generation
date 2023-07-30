## Unconditioned Training

- To train any ByGPT5 model, use run_clm_bygpt5.py. The below examples finetunes the English medium
  version of ByGPT5. 
  ```
  python run_clm_bygpt5.py --model_name nllg/bygpt5-medium-en \
  --train_file path_to_train_file \
  --output_dir path_to_trained_models/bygpt5-medium/en \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 16 \
  --save_steps 100000 \
  --validation_split_percentage 10 \
  --block_size 512 \
  --logging_steps 200 \
  --num_train_epochs 10
  ```
- To train GPT2 or GPTneo models, use run_clm.py. The below example finetunes the German large version
  of GPT2.
  ```
  python run_clm.py --model_name_or_path benjamin/gerpt2-large \
  --train_file path_to_train_file \
  --output_dir path_to_trained_models/gpt2-large/de \ 
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 16 \
  --save_steps 100000 \
  --validation_split_percentage 10 \ 
  --block_size 128 \
  --logging_steps 200 \ 
  --num_train_epochs 10 \
  --do_train \
  --do_eval \
