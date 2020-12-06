import os
import torch
from transformers import GPT2Tokenizer, GPT2Model

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# tokenizer.tokenize
# tokenize.encode
# tokenize.forward
tokenizer = GPT2Tokenizer.from_pretrained('gpt2', pad_token="[PAD]", additional_special_tokens=["[OBJ]"])
model = GPT2Model.from_pretrained('gpt2')
embedding_layer = model.resize_token_embeddings(len(tokenizer))  # Update the model embeddings with the new vocabulary size
inputs = tokenizer("who is the writer for [OBJ]", max_length=50, padding="max_length", truncation=True, return_tensors='pt')
outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", additional_special_tokens=["[OBJ]"])
inputs = tokenizer("who is the writer for [OBJ]", max_length=10, padding="max_length", truncation=True, return_tensors='pt')
