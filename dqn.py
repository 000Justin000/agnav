import torch
import torch.nn as nn
import utils

ndim_action = 32
ndim_state = 32

#---------------------------------------------------------------------
# get the knowledge graph
#---------------------------------------------------------------------
G = utils.read_MetaQA_KG()
#---------------------------------------------------------------------
# action embeddings
#---------------------------------------------------------------------
# find unique actions
action_set = set([edge[2]["type"] for edge in G.edges(data=True)])
action_set.add("terminate")
actions = sorted(list(action_set))
# create action embeddings
action_to_ix = dict(map(reversed, enumerate(actions)))
action_embeddings = nn.Embedding(len(actions), ndim_action)
#---------------------------------------------------------------------

# action value function
Qsa = nn.Sequential(nn.Linear(ndim_action+ndim_state,1), nn.Sigmoid())

# state transation function
dec = nn.LSTM(ndim_action, ndim_state)


