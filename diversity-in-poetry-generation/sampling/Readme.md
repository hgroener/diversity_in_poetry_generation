# Sampling

## Deep-speare and Structured-Adversary
For model inference, see [Deep-speare](https://github.com/b3nji87/deepspeare-fork) and [Structured-Adversary](https://github.com/b3nji87/Structured-Adversary-Fork). To process raw samples, run

```
python process_deepspeare_samples.py --raw_sample_path path_to_text_file \
--lang lang \
--save_path path_to_save_folder \
--save_name name_of_saved_sample \
--temperature temp_used_during_inference
```

and 

```
python process_structured_adversary_samples.py --raw_sample_path path_to_text_file \
--lang lang \
--save_path path_to_save_folder \
--save_name name_of_saved_sample \
--temperature temp_used_during_inference
```

## Unconditioned and style-conditioned LLMs
To generate a (vanilla) sample for a trained LLM, run

```
python sample_generation_llm.py --lang lang \
--trained_model_path trained_model_path \
--model_name model_name \
--out_path path_to_store_sample \
```
Refer to sample_generation_llm.py to get a list of available sampling techniques.

