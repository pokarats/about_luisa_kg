#!/usr/bin/env python
# coding: utf-8

# Based on the Colab notebook: GPT2 Language Model Fine-tuning with Texts from Shakespeare
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fcakyon/gpt2-shakespeare/blob/main/gpt2-shakespeare.ipynb)

# Install requirements
#!pip install -U transformers datasets torch sentencepiece pyyaml

# ## 1. Initialize Model and Tokenizer

# import the necessary libraries

import torch
import math
from transformers import GPT2Tokenizer, GPT2LMHeadModel, HfArgumentParser, TrainingArguments, Trainer, default_data_collator
from datasets import load_dataset, Dataset, DatasetDict
import pathlib

# - Initialize a GPT2 model with a language modelling head:
model = GPT2LMHeadModel.from_pretrained('gpt2')

# - Initialize GPT2 tokenizer:
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model.resize_token_embeddings(len(tokenizer))

# ## 2. Initialize Dataset
data_file = pathlib.Path(__file__).parent / "corpus_gpt2_final.txt"
print("Loading from file:", data_file)
with open(data_file) as fin:
    total_lines = [line.rstrip('\n') for line in fin]
print('Loaded all files, num total lines:', len(total_lines))
# create a Dataset object
dataset = Dataset.from_dict({'text': total_lines})

# split the data into train and test
test_and_train_dsd = dataset.train_test_split(train_size=0.8)
train_ds = test_and_train_dsd['train']
# split the test set into test and validation set
test_and_validation_dsd = test_and_train_dsd['test'].train_test_split(train_size=0.5)
test_ds = test_and_validation_dsd['test']
validation_ds = test_and_validation_dsd['train']

def join_dataset(ds):
    return Dataset.from_dict({'text': ['\n'.join(ds['text'])]})

train_ds = join_dataset(train_ds)
test_ds = join_dataset(test_ds)
validation_ds = join_dataset(validation_ds)

datasets = DatasetDict({'test': test_ds, 'train': train_ds, 'validation': validation_ds})

# - Tokenize all the texts:
column_names = datasets["train"].column_names
text_column_name = "text" if "text" in column_names else column_names[0]

def tokenize_function(examples):
    # truncate dataset with max accepted size of the model
    output = tokenizer(examples[text_column_name])
    return output

# tokenize dataset
tokenized_datasets = datasets.map(
    tokenize_function,
    batched=True,
    remove_columns=column_names,
    desc="Running tokenizer on dataset",
)

# - Split whole dataset into smaller sets of blocks:
# get block size (max input length of the model)
block_size = tokenizer.model_max_length
print(tokenizer.model_max_length)
if block_size > 1024:
    block_size = 1024

def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

# split total dataset into smaller sets of length block_size
lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    desc=f"Grouping texts in chunks of {block_size}",
)

train_dataset = lm_datasets["train"]
eval_dataset = lm_datasets["validation"]

# ## 3. Initialize Trainer
training_args = TrainingArguments(output_dir="output/", per_device_train_batch_size=1, num_train_epochs=30,
                                  save_total_limit=1)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    # Data collator will default to DataCollatorWithPadding, so we change it.
    data_collator=default_data_collator,
)

# # 4. Perform Training
# perform training
train_result = trainer.train()

# saves the tokenizer
trainer.save_model()

# save training metrics
metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)

# save training state
trainer.save_state()

# # 5. Evaluate Model
# perform evaluation over validation data
metrics = trainer.evaluate()

# calculate perplexity
try:
    perplexity = math.exp(metrics["eval_loss"])
except OverflowError:
    perplexity = float("inf")

# save perplexity
metrics["perplexity"] = perplexity
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)

# # 6. Generate Samples
# fix seed
torch.manual_seed(2)

# generate a text given prompt
def generate_text(input_text):
    # tokenize start of a sentence
    ids = tokenizer.encode(input_text,
                           return_tensors='pt').cuda()

    # generate samples by top-p sampling
    sample_output = model.generate(
        ids,
        do_sample=True,
        max_length=200,
        top_p=0.92,
        top_k=0,
        temperature=0.2
    )
    return tokenizer.decode(sample_output[0], skip_special_tokens=True)

def main():
    input = 'What inspires your art? A lot, experiences, texts, songs, films, conversations, observations, the daily life as news.'
    output = generate_text(input)
    # print generated texts
    print("Output:\n" + 100 * '-')
    print(output)

    # load the dataset with question-short answer pairs, generate extensions for them and save it in a txt document
    with open(ROOT_DIR + "/qa.txt", "r") as fin, open(ROOT_DIR + "/gpt2.txt", "w") as gpt2_output:
        for qa in fin:
            qa = qa.strip()
            # add dot at the end of the short answer
            if not qa.endswith("."):
                qa = qa + "."
            output = generate_text(qa)
            # output only the generated text, not includeng the question and short answer
            output = output[len(qa) + 1:].strip()

            # postprocess
            # replace new line with white space, so that there are no new lines within the generated output
            output = output.replace("\n", " ")
            # to cut the generated output after the last dot to keep it complete
            output = output[:output.rfind(".") + 1]
            # write the generated text to the file
            print(output, file=gpt2_output)
if __name__ == "__main__":
    main()
