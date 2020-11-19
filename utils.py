import networkx as nx
import pandas as pd
import numpy as np
import re
import torch
from collections import namedtuple
from transformers import AutoTokenizer, AutoModel

def read_MetaQA_KG():

    def edge2prefix(edge):
        if edge == "directed_by":
            return "director: "
        elif edge == "written_by":
            return "writer: "
        elif edge == "starred_actors":
            return "actor: "
        elif edge == "release_year":
            return "year: "
        elif edge == "in_language":
            return "language: "
        elif edge == "has_tags":
            return "tag: "
        elif edge == "has_genre":
            return "genre: "
        elif edge == "has_imdb_votes":
            return "votes: "
        elif edge == "has_imdb_rating":
            return "rating: "
        else:
            raise Exception("unexpected edge type \"" + edge + "\"")

    df = pd.read_csv("datasets/MetaQA/kb.txt", delimiter='|', names=["head", "edge", "tail"])

    decorated_heads = "movie: " + df["head"]
    decorated_tails = df["edge"].apply(edge2prefix) + df["tail"]
    fwd_edges = "fwd_"+df["edge"]
    rvs_edges = "rvs_"+df["edge"]

    G = nx.MultiDiGraph()
    G.add_nodes_from(zip(decorated_heads.unique(), [{"type": decorated_head.split(':')[0]} for decorated_head in decorated_heads.unique()]))
    G.add_nodes_from(zip(decorated_tails.unique(), [{"type": decorated_tail.split(':')[0]} for decorated_tail in decorated_tails.unique()]))
    G.add_edges_from(zip(decorated_heads, decorated_tails, [{"type": fwd_edge} for fwd_edge in fwd_edges]))
    G.add_edges_from(zip(decorated_tails, decorated_heads, [{"type": rvs_edge} for rvs_edge in rvs_edges]))

    return G

# def read_MetaQA_QA(question_type="1-hop"):
entity_token = "[unused0]"

def process_question(question):
    processed_question = re.sub(r"(\[.+\])", entity_token, question)
    entity = re.search(r"\[(.+)\]", question).group(1)

    return processed_question, entity

def process_answers(answers):
    return set(answers.split('|'))

question_type = "1-hop"
qa_train_texts = pd.read_csv("datasets/MetaQA/"+question_type+"/vanilla/qa_train.txt", delimiter='\t', names=["question", "answers"])
qa_train_qtype = pd.read_csv("datasets/MetaQA/"+question_type+"/qa_train_qtype.txt", names=["question_type"])
qa_train = pd.concat([qa_train_texts, qa_train_qtype], axis=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", additional_special_tokens=[entity_token])

model = AutoModel.from_pretrained("bert-base-uncased")
processed_question, entity = process_question(qa_train["question"][0])
inputs = tokenizer(processed_question, return_tensors="pt")
output = model(**inputs)










