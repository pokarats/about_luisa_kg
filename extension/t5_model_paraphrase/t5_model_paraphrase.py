#!/usr/bin/env python
# coding: utf-8

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)

model = T5ForConditionalGeneration.from_pretrained('ramsrigouthamg/t5_paraphraser')
tokenizer = T5Tokenizer.from_pretrained('ramsrigouthamg/t5_paraphraser')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("device ", device)
model = model.to(device)


def paraphrase(sentence):
    sentence = sentence.strip()
    text = "paraphrase: " + sentence + " </s>"
    max_len = 256
    # encode the sentence
    # encoding = tokenizer.encode_plus(text, pad_to_max_length=True, return_tensors="pt")
    encoding = tokenizer.encode_plus(text, return_tensors="pt")
    input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)
    # set top_k = 50 and set top_p = 0.95 and num_return_sequences = 3
    beam_outputs = model.generate(
        input_ids=input_ids, attention_mask=attention_masks,
        do_sample=True,
        max_length=256,
        top_k=120,
        top_p=0.98,
        early_stopping=True,
        num_return_sequences=10)

    final_outputs = []
    for beam_output in beam_outputs:
        sent = tokenizer.decode(beam_output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        if sent.lower() != sentence.lower() and sent not in final_outputs:
            final_outputs.append(sent)
    return final_outputs[0]


def main():
    # paraphrase tf_idf output corpus
    with open("data/tf_idf_output.txt", "r") as tf_idf_output, open("results/t5_paraphrase_new.txt", "w") as t5_output:
        # with open(ROOT_DIR + "/gpt2_output.txt", "r") as tf_idf_output, open(ROOT_DIR + "/gpt_2_t5_paraphrase.txt",
        # "w") as t5_output:
        paraphrase_list = []
        for sentence in tf_idf_output:
            paraphrased_sentence = paraphrase(sentence)
            paraphrase_list.append(paraphrased_sentence)
            print(paraphrased_sentence, file=t5_output)


if __name__ == "__main__":
    main()
