import multiprocessing
import re
from typing import (
    List,
    Dict,
    Callable,
)

import evaluate
import numpy as np
from datasets import (
    load_dataset,
    get_dataset_split_names
)
from promptsource.templates import (
    DatasetTemplates,
    Template
)
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    DataCollatorForSeq2Seq,
    BloomForCausalLM,
    TrainingArguments,
    Trainer
)

## Huggingface Datasets
DATASET_NAME: str = "xsum"
DATASET_TRAIN_NUM_ROWS: int = 204045      # Number of rows in the original train dataset
DATASET_STREAMING: bool = True                    # If using Dataset streaming
DATASET_TRAIN_NUM_SELECT: int = 2048       # Number of rows to use for training
DATASET_VALIDATE_NUM_SELECT: int = 128

# Huggingface Tokenizer (BLOOM default token length is 2048)
MAX_TOKEN_LENGTH: int = 512         # Max token length to avoid out of memory
PER_DEVICE_BATCH_SIZE: int = 1       # GPU batch size

# Huggingface Model
MODEL = "bigscience/bloom-560m"

# Training
NUM_EPOCHS: int = 3
MAX_STEPS: int = NUM_EPOCHS * DATASET_TRAIN_NUM_SELECT if DATASET_STREAMING else -1

train = load_dataset("xsum", split="train", streaming=DATASET_STREAMING)

prompt_templates = DatasetTemplates( dataset_name=DATASET_NAME)
template: Template = prompt_templates['summarize_DOC']

# # Preprocess
tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)


def get_convert_to_request_response(template: Template) -> Callable:
    def _convert_to_prompt_response(example: Dict[str, str]) -> Dict[str, str]:
        """Generate prompt, response as a dictionary:
        {
            "prompt": "Summarize: ...",
            "response": "..."
        }

        NOTE: DO NOT use with dataset map function( batched=True). Use batch=False

        Args:
            example: single {document, summary} pair to be able to apply template
        Returns: a dictionary of pro
        """
        # assert isinstance(example, dict), f"expected dict but {type(example)}.\n{example}"
        assert isinstance(example['document'], str), f"expected str but {type(example['document'])}."
        prompt, response = template.apply(example=example, truncate=False)
        return {
            "prompt": re.sub(r'[\s\'\"]+', ' ', prompt),
            "response": re.sub(r'[\s\'\"]+', ' ', response)
        }

    return _convert_to_prompt_response


convert_to_request_response: Callable = get_convert_to_request_response(template=template)


def tokenize_prompt_response(examples):
    """Generate the model inputs in the dictionary with format:
    {
        "input_ids": List[int], 
        "attention_mask": List[int]",
        "labels": List[int]
    }
    
    Note: Huggngface dataaset map(batched=True, batch_size=n) merges values of 
    n dictionarys into a values of the key. If you have n instances of {"key", "v"}, then
    you will get {"key": ["v", "v", "v", ...] }.
    
    Args:
        examples:   a dictionary of format {
            "prompt": [prompt+],
            "response": [respnse+]
        } where + means more than one instance because of Dataset.map(batched=True)
    """    
    inputs: Dict[str, List[int]] = tokenizer(
        text_target=examples["prompt"], 
        max_length=MAX_TOKEN_LENGTH, 
        truncation=True
    )

    labels: Dict[str, List[int]] = tokenizer(
        text_target=examples["response"], 
        max_length=MAX_TOKEN_LENGTH, 
        truncation=True,
        padding='max_length',
    )
    inputs["labels"] = labels["input_ids"]
    
    return inputs


remove_column_names: List[str] = list(train.features.keys())
tokenized_train = train.map(
    function=convert_to_request_response, 
    batched=False,
    batch_size=2048,
    drop_last_batch=False,
    remove_columns=remove_column_names,
).map(
    function=tokenize_prompt_response, 
    batched=True,
    batch_size=32,
    drop_last_batch=True,
    remove_columns=['prompt', 'response']
).shuffle(
    seed=42
).with_format(
    "torch"
)

if DATASET_STREAMING:
    tokenized_train = tokenized_train.take(DATASET_TRAIN_NUM_SELECT)
else:
    tokenized_train = tokenized_train.select(
        indices=range(DATASET_TRAIN_NUM_SELECT)
    )

del train


tokenized_validation =  load_dataset(
    path="xsum", 
    split="validation", 
    streaming=DATASET_STREAMING
).map(
    function=convert_to_request_response, 
    batched=False,
    batch_size=2048,
    drop_last_batch=False,
    remove_columns=remove_column_names,
).map(
    function=tokenize_prompt_response, 
    batched=True,
    batch_size=32,
    drop_last_batch=True,
    remove_columns=['prompt', 'response']
).with_format(
    "torch"
)

if DATASET_STREAMING:
    tokenized_validation = tokenized_validation.take(DATASET_TRAIN_NUM_SELECT)
else:
    tokenized_validation = tokenized_validation.select(
        indices=range(DATASET_TRAIN_NUM_SELECT)
    )

# # Training
model = BloomForCausalLM.from_pretrained(MODEL)
model.cuda()


def predict(prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors='pt')
    print(inputs["input_ids"].shape)
    
    response_tokens = model.generate(
        inputs["input_ids"].cuda(), 
        max_new_tokens=1,
        do_sample=False, 
        top_k=50, 
        top_p=0.9
    )[0]
    response = tokenizer.decode(response_tokens, skip_special_tokens=True)
    return response


# DataCollatorWithPadding does not pad 'labels' which causes an error at train()
# https://stackoverflow.com/a/74228547/4281353
data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer, 
    padding='max_length',
    pad_to_multiple_of=8,
    max_length=MAX_TOKEN_LENGTH,
    return_tensors='pt'
)

# ## Evaluation
rouge = evaluate.load("rouge")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}


# ## Trainer API
training_args = TrainingArguments(
    output_dir="bloom_finetuned",
    max_steps=MAX_STEPS,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
    per_device_eval_batch_size=PER_DEVICE_BATCH_SIZE,
    learning_rate=2e-5,
    weight_decay=0.01, 
    fp16=True,
    no_cuda=False,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=3,
    log_level="debug",
    disable_tqdm=False,
    push_to_hub=False,
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_validation,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)


trainer.train()
