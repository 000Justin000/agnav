import networkx as nx
import pandas as pd
import random
import re
import torch
from collections import namedtuple
from transformers import AutoTokenizer

QAInstance = namedtuple("QAInstance", ["question", "tokenized_inputs", "decorated_entity", "answer_set"])
Episode = namedtuple("Episode", ["qa_instance", "kgnode_chain", "action_chain", "reward_chain"])

def unique(items):
    return sorted(list(set(items)))


def read_MetaQA_KG():

    def edge_to_prefix(edge):
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
    decorated_tails = df["edge"].apply(edge_to_prefix) + df["tail"]
    fwd_edges = "fwd_"+df["edge"]
    rvs_edges = "rvs_"+df["edge"]

    G = nx.MultiDiGraph()
    G.add_nodes_from(zip(decorated_heads.unique(), [{"type": decorated_head.split(':')[0]} for decorated_head in decorated_heads.unique()]))
    G.add_nodes_from(zip(decorated_tails.unique(), [{"type": decorated_tail.split(':')[0]} for decorated_tail in decorated_tails.unique()]))
    G.add_edges_from(zip(decorated_heads, decorated_tails, [{"type": fwd_edge} for fwd_edge in fwd_edges]))
    G.add_edges_from(zip(decorated_tails, decorated_heads, [{"type": rvs_edge} for rvs_edge in rvs_edges]))

    return G


def read_MetaQA_Instances(question_type="1-hop", device="cpu"):
    entity_token = "[unused0]"
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", additional_special_tokens=[entity_token])

    def process_question(question):
        #---------------------------------------------------------------------------------
        processed_question = re.sub(r"(\[.+\])", entity_token, question)
        entity = re.search(r"\[(.+)\]", question).group(1)
        #---------------------------------------------------------------------------------
        return processed_question, entity

    def process_answers(answers):
        return set(answers.split('|'))

    def info_to_instance(info):
        #---------------------------------------------------------------------------------
        processed_question, entity = process_question(info["question"])
        #---------------------------------------------------------------------------------
        tokenized_inputs = tokenizer(processed_question, return_tensors="pt")
        tokenized_inputs["input_ids"] = tokenized_inputs["input_ids"].to(device)
        tokenized_inputs["token_type_ids"] = tokenized_inputs["token_type_ids"].to(device)
        tokenized_inputs["attention_mask"] = tokenized_inputs["attention_mask"].to(device)
        #---------------------------------------------------------------------------------
        decorated_entity = info["question_type"].split('_')[0] + ": " + entity
        #---------------------------------------------------------------------------------
        answer_set = process_answers(info["answers"])
        #---------------------------------------------------------------------------------
        return QAInstance(info["question"], tokenized_inputs, decorated_entity, answer_set)

    #-------------------------------------------------------------------------------------
    qa_text_train = pd.read_csv("datasets/MetaQA/"+question_type+"/vanilla/qa_train.txt", delimiter='\t', names=["question", "answers"])
    qa_qtype_train = pd.read_csv("datasets/MetaQA/"+question_type+"/qa_train_qtype.txt", names=["question_type"])
    qa_info_train = pd.concat([qa_text_train, qa_qtype_train], axis=1)
    qa_instance_train = qa_info_train.apply(info_to_instance, axis=1)
    #-------------------------------------------------------------------------------------
    qa_text_dev = pd.read_csv("datasets/MetaQA/"+question_type+"/vanilla/qa_dev.txt", delimiter='\t', names=["question", "answers"])
    qa_qtype_dev = pd.read_csv("datasets/MetaQA/"+question_type+"/qa_dev_qtype.txt", names=["question_type"])
    qa_info_dev = pd.concat([qa_text_dev, qa_qtype_dev], axis=1)
    qa_instance_dev = qa_info_dev.apply(info_to_instance, axis=1)
    #-------------------------------------------------------------------------------------
    qa_text_test = pd.read_csv("datasets/MetaQA/"+question_type+"/vanilla/qa_test.txt", delimiter='\t', names=["question", "answers"])
    qa_qtype_test = pd.read_csv("datasets/MetaQA/"+question_type+"/qa_test_qtype.txt", names=["question_type"])
    qa_info_test = pd.concat([qa_text_test, qa_qtype_test], axis=1)
    qa_instance_test = qa_info_test.apply(info_to_instance, axis=1)
    #-------------------------------------------------------------------------------------

    return qa_instance_train, qa_instance_dev, qa_instance_test


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, episode):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = episode
        self.position = (self.position + 1) % self.capacity

    def sample_random(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        return batch

    def sample_last(self, batch_size):
        pointer = self.position
        batch = []
        for _ in range(batch_size):
            pointer = (pointer - 1 + self.capacity) % self.capacity
            batch.append(self.memory[pointer])
        return batch

    def __len__(self):
        return len(self.memory)
