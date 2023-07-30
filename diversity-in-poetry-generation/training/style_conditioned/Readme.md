## Style-conditioned Training

To train a given LLM in a style-conditioned manner, use poetry_training.py. The below example finetunes the English small version of GPTneo:
```
python poetry_training.py \
--model_name_or_path EleutherAI/gpt-neo-125m \
--out_dir path_to_trained_models \
--out_name poetry-gptneo-small \
--lang en

```
