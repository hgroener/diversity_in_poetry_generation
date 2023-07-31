## Unconditioned Training
- We train LLMs in an uncondtioned manner using a text file containing raw quatrains from QuaTrain as training data. See training_data/parse_quatrain.py for deriving such files.

- To train any ByGPT5 model, use run_clm_bygpt5.py. This file is inspired by run_clm.py (see below). The following examples finetunes the English medium version of ByGPT5 obtained from the hugging face model hub. 
  ```
  python run_clm_bygpt5.py --model_name nllg/bygpt5-medium-en \
  --train_file path_to_train_file \
  --output_dir path_to_trained_model_directory \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 16 \
  --save_steps 100000 \
  --validation_split_percentage 10 \
  --block_size 512 \
  --logging_steps 200 \
  --num_train_epochs 10
  ```
  
- To train GPT2 or GPTneo models, use run_clm.py. This training file is taken from [the official tranfsormers repository](https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling). The below example finetunes the German large version of GPT2 obtained from the hugging face model hub.
  ```
  python run_clm.py --model_name_or_path benjamin/gerpt2-large \
  --train_file path_to_train_file \
  --output_dir path_to_trained_model_directory \ 
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 16 \
  --save_steps 100000 \
  --validation_split_percentage 10 \ 
  --block_size 128 \
  --logging_steps 200 \ 
  --num_train_epochs 10 \
  --do_train \
  --do_eval \
