import transformers
from transformers import Trainer, TrainingArguments, default_data_collator
from uniformers.models.bygpt5 import ByGPT5LMHeadModel, ByGPT5Tokenizer

from datasets import load_dataset
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from itertools import chain

import math
import evaluate
import logging
import sys
import datasets
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str)
parser.add_argument("--train_file", type=str)
parser.add_argument("--output_dir", type=str)
parser.add_argument("--validation_split_percentage", type=int, default=10)
parser.add_argument("--block_size", type=int, default=1000)
parser.add_argument("--logging_steps", type=float, default=200.0)
parser.add_argument("--save_steps", type=int, default=1000)
parser.add_argument("--per_device_train_batch_size", type=int, default=8)
parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
parser.add_argument("--num_train_epochs", type=float, default=3.0
)
args = parser.parse_args()

training_args = TrainingArguments(
    output_dir=args.output_dir,
    #warmup_steps=100,
    logging_steps=args.logging_steps,
    save_steps=args.save_steps,
    per_device_train_batch_size = args.per_device_train_batch_size,
    per_device_eval_batch_size = args.per_device_eval_batch_size,
    do_train=True,
    do_eval=True,
    #gradient_accumulation_steps = 8,
    #learning_rate = 0.00002,
    num_train_epochs = args.num_train_epochs,
    #weight_decay=0.01,
    resume_from_checkpoint=True
)

logger = logging.getLogger(__name__)
# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

log_level = training_args.get_process_log_level()
logger.setLevel(log_level)
datasets.utils.logging.set_verbosity(log_level)
transformers.utils.logging.set_verbosity(log_level)
transformers.utils.logging.enable_default_handler()
transformers.utils.logging.enable_explicit_format()

# Log on each process the small summary:
logger.warning(
    f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
    + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
)

logger.info(f"Training/evaluation parameters {training_args}")

model = ByGPT5LMHeadModel.from_pretrained(args.model_name)
tokenizer = ByGPT5Tokenizer.from_pretrained(args.model_name)


last_checkpoint = None
if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
    last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. "
            "Use --overwrite_output_dir to overcome."
        )
    elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(
            f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
            "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
        )

data_files = {}
dataset_args = {}
data_files["train"] = args.train_file

extension = "text"
dataset_args["keep_linebreaks"] = True
raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            **dataset_args,
)

raw_datasets["validation"] = load_dataset(
            extension,
            data_files=data_files,
            split=f"train[:{args.validation_split_percentage}%]",
            **dataset_args,
)

raw_datasets["train"] = load_dataset(
            extension,
            data_files=data_files,
            split=f"train[{args.validation_split_percentage}%:]",
            **dataset_args,
)

embedding_size = model.get_input_embeddings().weight.shape[0]
if len(tokenizer) > embedding_size:
    model.resize_token_embeddings(len(tokenizer))

column_names = list(raw_datasets["train"].features)
text_column_name = "text" if "text" in column_names else column_names[0]

tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

def tokenize_function(examples):
    with CaptureLogger(tok_logger) as cl:
        output = tokenizer(examples[text_column_name])
    # clm input could be much much longer than block_size
    if "Token indices sequence length is longer than the" in cl.out:
        tok_logger.warning(
            "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
            " before being passed to the model."
        )
    return output

tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                remove_columns=column_names,
                load_from_cache_file= True,
                desc="Running tokenizer on dataset",
            )

def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= args.block_size:
        total_length = (total_length // args.block_size) * args.block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + args.block_size] for i in range(0, total_length, args.block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    load_from_cache_file=True,
    desc=f"Grouping texts in chunks of {args.block_size}",
)

train_dataset = lm_datasets["train"]
eval_dataset = lm_datasets["validation"]

def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
    # Depending on the model and config, logits may contain extra tensors,
    # like past_key_values, but logits always come first
        logits = logits[0]
    return logits.argmax(dim=-1)

metric = evaluate.load("accuracy")

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # preds have the same shape as the labels, after the argmax(-1) has been calculated
    # by preprocess_logits_for_metrics but we need to shift the labels
    labels = labels[:, 1:].reshape(-1)
    preds = preds[:, :-1].reshape(-1)
    return metric.compute(predictions=preds, references=labels)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    # Data collator will default to DataCollatorWithPadding, so we change it.
    data_collator=default_data_collator,
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
)

checkpoint = None
checkpoint = training_args.resume_from_checkpoint
if last_checkpoint is not None:
    checkpoint = last_checkpoint


train_result = trainer.train()
trainer.save_model()

metrics = train_result.metrics

max_train_samples = len(train_dataset)
metrics["train_samples"] = min(max_train_samples, len(train_dataset))

trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()

logger.info("*** Evaluate ***")

metrics = trainer.evaluate()

max_eval_samples = len(eval_dataset)
metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
try:
    perplexity = math.exp(metrics["eval_loss"])
except OverflowError:
    perplexity = float("inf")
metrics["perplexity"] = perplexity

trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)

kwargs = {"finetuned_from": args.model_name, "tasks": "text-generation"}
kwargs["dataset_tags"] = 'QuaTrain'
kwargs["dataset"] = 'QuaTrain'
