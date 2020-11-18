import torch
from transformers import AutoTokenizer, AutoModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased").to(device)

inputs = tokenizer("Hello world!", return_tensors="pt")
outputs = model(input_ids=inputs["input_ids"].to(device), token_type_ids=inputs["token_type_ids"].to(device), attention_mask=inputs["attention_mask"].to(device))
