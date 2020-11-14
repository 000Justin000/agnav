import networkx as nx
import pandas as pd
import numpy as np

def read_MetaQA_KB():

    def edge2prefix(edge):
        if edge in ["directed_by", "written_by", "starred_actors"]:
            return "Person: "
        elif edge == "release_year":
            return "Year: "
        elif edge == "in_language":
            return "Language: "
        elif edge == "has_tags":
            return "Tag: "
        elif edge == "has_genre":
            return "Genre: "
        elif edge == "has_imdb_votes":
            return "Votes: "
        elif edge == "has_imdb_rating":
            return "Rating: "
        else:
            raise Exception("unexpected edge type \"" + edge + "\"")

    df = pd.read_csv("datasets/MetaQA/kb.txt", delimiter='|', names=["head", "edge", "tail"])

    decorated_heads = "Movie: " + df["head"]
    decorated_tails = df["edge"].apply(edge2prefix) + df["tail"]
    fwd_edges = "fwd_"+df["edge"]
    rvs_edges = "rvs_"+df["edge"]

    G = nx.MultiDiGraph()
    G.add_nodes_from(zip(decorated_heads.unique(), [{"type": decorated_head.split(':')[0]} for decorated_head in decorated_heads.unique()]))
    G.add_nodes_from(zip(decorated_tails.unique(), [{"type": decorated_tail.split(':')[0]} for decorated_tail in decorated_tails.unique()]))
    G.add_edges_from(zip(decorated_heads, decorated_tails, [{"type": fwd_edge} for fwd_edge in fwd_edges]))
    G.add_edges_from(zip(decorated_tails, decorated_heads, [{"type": rvs_edge} for rvs_edge in rvs_edges]))

    return G

G = read_MetaQA_KB()

qa_1 = pd.read_csv("datasets/MetaQA/1-hop/vanilla/qa_train.txt", delimiter='\t', names=["question", "answers"])