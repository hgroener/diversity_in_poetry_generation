import pickle
from mosestokenizer import *
from huggingface_hub import login, notebook_login
import torch
from argparse import ArgumentParser
from datasets import load_dataset, Dataset
import pandas as pd
from typing import *
import gc
import pickle
import os
from transformers import EarlyStoppingCallback, TrainerCallback
#os.environ['CUDA_VISIBLE_DEVICES']="0,1"

print("gpu available: ", torch.cuda.is_available())
device = 'cuda' if torch.cuda.is_available() else 'cpu'
access_token = "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"

login(token=access_token)

from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, AutoConfig, AutoTokenizer, BitsAndBytesConfig, pipeline, logging
from transformers import DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

logging.set_verbosity(logging.CRITICAL)

l2l = {
        'en': 'English',
        'de': 'German'
    }


def load_model(model_name="meta-llama/Llama-2-7b-hf", instruct=False, conditioned=False):
    compute_dtype = getattr(torch, "float16")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if 'llama' in model_name:
        quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                          bnb_4bit_compute_dtype=compute_dtype, bnb_4bit_use_double_quant=False)
        model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quant_config, device_map='auto')
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto')
    print("\nModel is on cuda: ", next(model.parameters()).is_cuda)
    print(model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    #Todo: maybe add special tokens


    #if 'gpt' in model_name:
    if not instruct:
        #special_tokens_dict = {'additional_special_tokens': ['<M>', '<Q>', '<R>', '<A>', '<eos>']}
        #special_tokens_dict = {'additional_special_tokens': ['<M>', '<R>', '<A>', '<eos>']}
        #tokenizer.add_special_tokens(special_tokens_dict)
        #tokenizer.add_tokens(['<eos>', '<q>', '</q>'])
        tokenizer.add_tokens(['<quatrain>', '</quatrain>'])
        #tokenizer.add_tokens(['<line1>', '<line2>', '<line3>', '<line4>'])
        #tokenizer.add_tokens(['<line>'])
        #tokenizer.add_tokens(['<line>'])
        #tokenizer.add_tokens(['<M>', '<R>', '<A>', '<eos>'])
        if conditioned:
            special_tokens_dict = {'additional_special_tokens': ['[ABCD]', '[AABC]', '[ABAC]', '[ABCC]', '[ABBA]',
                                  '[ABAB]', '[ABCB]', '[ABBC]', '[ABBB]', '[AABA]',
                                  '[AABB]', '[AAAB]', '[ABCA]', '[AAAA]', '[ABAA]',
                                  '<iambus>', '<trochee>', '<anapaest>', '<dactyl>', '<other>', '<amphibrach>', '<alexandrine>']}
            tokenizer.add_special_tokens(special_tokens_dict)
        model.resize_token_embeddings(len(tokenizer))

    return tokenizer, model


def get_train_args(args):
    training_params = TrainingArguments(output_dir=args.out_dir, num_train_epochs=args.epochs, per_device_train_batch_size=args.batch_size,
                                        gradient_accumulation_steps=args.gradient_accumulation_steps, optim="paged_adamw_32bit", save_strategy='epoch',
                                        logging_strategy='epoch', evaluation_strategy='epoch', learning_rate=args.learning_rate, weight_decay=0.001, fp16=False,
                                        bf16=False, max_grad_norm=0.3, max_steps=-1, warmup_ratio=0.03,
                                        group_by_length=True, lr_scheduler_type="cosine", report_to="tensorboard", disable_tqdm=False,
                                        save_total_limit=1, load_best_model_at_end=True, metric_for_best_model='eval_loss')
    if 'llama' in args.model_name:
        peft_params = LoraConfig(lora_alpha=16, lora_dropout=0.05, r=16, bias="none", task_type="CAUSAL_LM",
                                 target_modules=['q_proj', 'v_proj', 'embed_tokens', 'k_proj', 'o_proj'],
                                 modules_to_save=["embed_tokens"])
    else:
        peft_params = None
    return training_params, peft_params


def create_prompt_chat(df_row, lang: str, conditioned=False, detokenizer=None) -> str:
    #print(df_row)
    if detokenizer:
        quatrain = '\n'.join([detokenizer(l.split()) for l in list(df_row['text'])])
    else:
        quatrain = '\n'.join(list(df_row['text']))
    if conditioned:
        if df_row['alliteration'] < 0.05:
            alli = 'low'
        elif df_row['alliteration'] < 0.1:
            alli = 'medium'
        else:
            alli = 'high'
        prompt = f"<s>[INST] <<SYS>>\nYou are a creative poet.\n" \
                 f"<</SYS>>\n\nWrite a quatrain in {lang} with *rhyme* \"{df_row['rhyme']}\", *meter* \"{df_row['meter']}\", and \"{alli}\" *alliteration*.[/INST]" \
                 f"\n{quatrain}</s>"
    else:
        prompt = f"<s>[INST] <<SYS>>\nYou are a creative poet.\n" \
                 f"<</SYS>>\n\nWrite a quatrain in {lang}.[/INST]" \
                 f"\n{quatrain}</s>"
    return prompt

#BASIC_PROMPT = "Below is the quatrain:\n"
#BASIC_PROMPT = "<Q>"
BASIC_PROMPT = "Write a quatrain below:\n<quatrain>"
#PREFIX = f"[R] unknown [M] unknown [A] unknown\n"
#PREFIX = f"[R] unknown [M] unknown [A] unknown\n"
#PREFIX = tokenizer.eos_token * 3

def create_prompt(df_row, lang: str, conditioned=False, detokenizer=None, tokenizer=None) -> str:
    if detokenizer:
        #quatrain = f'\n'.join([detokenizer(l.split()).replace('\n', ' ') for l in list(df_row['text'])])
        #quatrain = f'\n'.join([detokenizer(l.split()).replace('\n', ' ') for l in list(df_row['text'])])
        #quatrain = '<eos>'.join([detokenizer(l.split()) for l in list(df_row['text'])])
        #quatrain = ''.join([token+line+'\n' for token, line in zip(['<line1>', '<line2>', '<line3>', '<line4>'], [detokenizer(l.split()) for l in list(df_row['text'])])])
        #quatrain = ''.join([token+line+'\n' for token, line in zip(['<eos>']*4, [detokenizer(l.split()) for l in list(df_row['text'])])])
        #quatrain = ''.join([line+token+'\n' for token, line in zip(['<eos>']*4, [detokenizer(l.split()) for l in list(df_row['text'])])])
        quatrain = ''.join([line+token+'\n' for token, line in zip(['']*4, [detokenizer(l.split()) for l in list(df_row['text'])])])
    else:
        quatrain = '\n'.join(list(df_row['text']))
    prompt = f'{tokenizer.bos_token}'
    if conditioned:
        if df_row['alliteration'] < 0.05:
            alli = 'low'
        elif df_row['alliteration'] < 0.1:
            alli = 'medium'
        else:
            alli = 'high'
        #prefix = f"[R] {df_row['rhyme']} [M] {df_row['meter']} [A] {alli}\n"
        #prefix = f"<R> {df_row['rhyme']} <M> {df_row['meter']} <A> {alli}\n"
        prefix = f"[{df_row['rhyme']}] <{df_row['meter']}>\n"
        #prefix = f"A quatrain has a rhyme scheme \"{df_row['rhyme']}\" and a {df_row['meter']} meter. "
        #prefix = tokenizer.eos_token.join([df_row['rhyme'], df_row['meter'], alli]) + tokenizer.eos_token

    else:
        #prompt += f"A quatrain is a rhymed grouping of four lines. Following is a quatrain:\n{quatrain}{tokenizer.eos_token}"
        #prompt += f"A poem of 4 lines:\n{quatrain}\n{tokenizer.eos_token}"
        #prompt += f"{BASIC_PROMPT}\n{quatrain}{tokenizer.eos_token}"
        #prompt += f"{BASIC_PROMPT}\n[Q] {quatrain}{tokenizer.eos_token}"
        prefix = ""
    prompt += prefix + BASIC_PROMPT + f"{quatrain.strip()}</quatrain>{tokenizer.eos_token}"
    #print(prompt)
    #raise ValueError
    return prompt


class CustomCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        #print(kwargs)
        #print(dir(args))
        model = kwargs['model']
        tokenizer = kwargs['tokenizer']
        model.eval()
        #PREFIX = (tokenizer.eos_token + " ")* 3
        #prompt = f"{tokenizer.bos_token}[QUATRAIN] " #if not args.instruct else f"<s>[INST] <<SYS>>\nYou are a creative poet.\n <</SYS>>\n\nWrite a quatrain in {l2l[args.lang]}.[/INST]\n"
        #prompt = f""
        #prompt = f"{tokenizer.bos_token}{PREFIX}{BASIC_PROMPT}"
        #prompt = tokenizer.bos_token+BASIC_PROMPT
        prompt = BASIC_PROMPT
        #prompt = f"A quatrain is a rhymed grouping of four lines. Following is an example:\n"
        pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, temperature=0.6,
                        num_return_sequences=1, do_sample=True, top_p=0.9, top_k=50, repetition_penalty=1.2, max_new_tokens=70, min_new_tokens=10)
        #pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_new_tokens=128, do_sample=False)
        print("*"*15+f"\nSample 10 outputs at epoch {state.epoch}")
        for i in range(10):
            while True:
                try:
                    results = pipe(prompt)
                    break
                except:
                    continue

            # for i, result in enumerate(results):
            print(f'\n---------Out {i + 1} of epoch {state.epoch}:')
            print(results[0]['generated_text'])
            print('---------Out ends.')
        print("*" * 30)


def load_data(path, lang, conditioned, tokenizer, instruct=False, mode='debug'):

    with open(path, 'rb') as f:
        df = pickle.load(f)
    if mode == 'debug':
        df = df.iloc[:1000]
    detokenizer = None if lang == 'de' else MosesDetokenizer('en')
    if instruct:
        df['text'] = df.apply(lambda x: create_prompt_chat(x, l2l[lang], conditioned, detokenizer), axis=1)
    else:
        df['text'] = df.apply(lambda x: create_prompt(x, l2l[lang], conditioned, detokenizer, tokenizer), axis=1)
    if detokenizer:
        detokenizer.close()

    dataset = Dataset.from_pandas(df)
    return dataset


def main(args):
    gc.collect()
    path = "{}/{}/{}_{}.pkl"

    tokenizer, model = load_model(args.model_name, args.instruct, args.conditioned)

    dataset = load_data(path.format(args.data_dir, args.lang, 'train', 100), args.lang, args.conditioned, tokenizer, args.instruct, args.mode)
    eval_dataset = load_data(path.format(args.data_dir, args.lang, 'devtest', 100), args.lang, args.conditioned, tokenizer, args.instruct, args.mode)
    print('\nData sample:\n', dataset['text'][0])

    training_params, peft_params = get_train_args(args)
    if 'llama' in args.model_name:
        model = get_peft_model(model, peft_params)
        print("\nTrainable parameters with lora:")
        model.print_trainable_parameters()

    '''
    if args.mode == 'debug':
        # prompt = "Write a quatrain."
        #prompt = f"<s>[INST] <<SYS>>\nYou are a creative poet.\n" \
        #         f"<</SYS>>\nWrite a quatrain in English.[/INST]"
        # = f"{tokenizer.bos_token} Write a quatrain in English.\n" if args.lang == 'en' else f"{tokenizer.bos_token} \n <quatrain> "
        prompt = f"{tokenizer.bos_token}[QUATRAIN] " if not args.instruct else f"<s>[INST] <<SYS>>\nYou are a creative poet.\n <</SYS>>\n\nWrite a quatrain in {l2l[args.lang]}.[/INST]\n"
        pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_new_tokens=70, temperature=0.6,
                        num_return_sequences=1, do_sample=True, top_p=0.9, top_k=50, repetition_penalty=1.2)
        for i in range(5):
            results = pipe(prompt)
        #for i, result in enumerate(results):
            print(f'\n---------Out {i + 1}:')
            print(results[0]['generated_text'])
            print('---------Out ends.')
    '''
    callbacks = [EarlyStoppingCallback(early_stopping_patience=3), CustomCallback()]
    print(callbacks)
    trainer = SFTTrainer(model=model, train_dataset=dataset, peft_config=peft_params, dataset_text_field="text",
                         max_seq_length=args.max_len, tokenizer=tokenizer, args=training_params, packing=False,
                         eval_dataset=eval_dataset, callbacks=callbacks)
    gc.collect()
    torch.cuda.empty_cache()
    trainer.train()

    model.eval()
    '''
    if 'gpt' in args.model_name:
        tokenizer.save_pretrained(os.path.join(args.out_dir, 'tokenizer'))
        model.save_pretrained(os.path.join(args.out_dir, 'model'))

    
    print('Sample 5 outputs...')
    #prompt = f"<s>[INST] <<SYS>>\nYou are a creative poet.\n" \
    #         f"<</SYS>>\nWrite a quatrain in English.[/INST]"
    #prompt = "<s> Write a quatrain. </s>" if args.lang == 'en' else "<s> Schreiben Sie ein Quartett. </s>"
    #prompt = f"{tokenizer.bos_token} Write a quatrain in English.\n" if args.lang == 'en' else f"{tokenizer.bos_token} \n <quatrain> "
    prompt = f"{tokenizer.bos_token}[QUATRAIN] " if not args.instruct else f"<s>[INST] <<SYS>>\nYou are a creative poet.\n <</SYS>>\n\nWrite a quatrain in {l2l[args.lang]}.[/INST]\n"
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_new_tokens=70, temperature=0.6,
                    num_return_sequences=1, do_sample=True, top_p=0.9, top_k=50, repetition_penalty=1.2)
    for i in range(5):
        results = pipe(prompt)
        # for i, result in enumerate(results):
        print(f'\n---------Out {i + 1}:')
        print(results[0]['generated_text'])
        print('---------Out ends.')
    '''
    print('Training complete!!!!')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-m', '--model_name', default="meta-llama/Llama-2-7b-hf", type=str)
    parser.add_argument('--out_dir', default="tmp", type=str)
    parser.add_argument('--data_dir', default="../data/train", type=str)
    parser.add_argument('--lang', default="en", type=str)
    parser.add_argument('-bs', '--batch_size', default=16, type=int)
    parser.add_argument('-lr', '--learning_rate', default=4e-05, type=float)
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int)
    parser.add_argument('--conditioned', action='store_true', help='Conditioned generation or not?')
    parser.add_argument('--instruct', action='store_true', help='If use inputs with instruct template? Set True only for llama for now.')
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--max_len', default=100, type=int)
    parser.add_argument('--mode', default='debug', type=str, help='debug or 100. debug: only using the first 100 samples from train/dev sets. 100: full data.')

    args = parser.parse_args()
    print(args)
    main(args)



