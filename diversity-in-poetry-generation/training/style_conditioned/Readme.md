## Style-conditioned Training

To train a given LLM in a style-conditioned manner, use poetry_training.py. This file is taken from the [Uniformers repository](https://github.com/potamides/uniformers). The below example finetunes the English small version of GPTneo obtained from the hugging face model hub:
```
python poetry_training.py \
--model_name_or_path EleutherAI/gpt-neo-125m \
--out_dir path_to_trained_model\
--out_name poetry-gptneo-small \
--lang en
```
