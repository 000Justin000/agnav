import random
import numpy as np
import re
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModel
from utils import read_MetaQA_KG, read_MetaQA_Instances

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ndim_state = 768
ndim_action = 64
epsilon = 0.3
gamma = 0.9
M = 1000
T = 3

#---------------------------------------------------------------------
# get the knowledge graph, question/answers instances
#---------------------------------------------------------------------
G = read_MetaQA_KG()
qa_train, qa_dev, qa_test = read_MetaQA_Instances()
#---------------------------------------------------------------------

#---------------------------------------------------------------------
# determine all possible actions
#---------------------------------------------------------------------
# find unique actions
possible_actions = sorted(list(set([edge[2]["type"] for edge in G.edges(data=True)])))
possible_actions.append("terminate")
action_to_ix = dict(map(reversed, enumerate(possible_actions)))
#---------------------------------------------------------------------

#---------------------------------------------------------------------
# trainable model parameters
#---------------------------------------------------------------------
# action_embeddings
emb = nn.Embedding(len(possible_actions), ndim_action).to(device)
# language embedding
enc = AutoModel.from_pretrained("bert-base-uncased").to(device)
# action value function
qsa = nn.Sequential(nn.Linear(ndim_action+ndim_state,1), nn.Sigmoid()).to(device)
# state transation function
dec = nn.RNNCell(ndim_action, ndim_state).to(device)
# container for the entire model
model = nn.ModuleList([emb,enc,qsa,dec])
#---------------------------------------------------------------------

optimizer = optim.Adam(model.parameters())
for m in range(M):
    tokenized_inputs, decorated_entity, answer_set = qa_train.sample(1).values[0]
    assert decorated_entity in G.nodes
    curr_node = decorated_entity

    # set initial values for state, action, and reward
    state = enc(**tokenized_inputs)[1]
    action = None
    reward = None

    for t in range(T):
        if curr_node != "termination":
            # available actions as going along one of the edges or terminate
            actions = list(set([info["type"] for (curr_node, next_node, info) in G.edges(curr_node, data=True)] + ["terminate"]))
            emb_actions = emb(torch.tensor([action_to_ix[action] for action in actions], dtype=torch.long).to(device))
            emb_sapairs = torch.cat((state.repeat(len(actions),1), emb_actions), 1)
            val_sapairs = qsa(emb_sapairs)

        if t != 0:
            if curr_node == "termination":
                reference = reward
            else:
                reference = reward + gamma*value_sa.max().item()

            # update the action-value function for the state and action at the previous step
            val_sapair0 = qsa(torch.cat((state,emb(torch.tensor([action_to_ix[action]], dtype=torch.long).to(device))), 1))
            optimizer.zero_grad()
            loss = (val_sapair0-reference)*(val_sapair0-reference)
            loss.backward()
            optimizer.step()

        # selection action with epsilon-greedy policy
        if random.random() < epsilon:
            action = random.choice(actions)
        else:
            action = actions[val_sapairs.argmax()]

        # take the action
        if action == "terminate":
            reward = 1.0 if (re.match(r".+: (.+)", curr_node).group(1) in answer_set) else 0.0
            curr_node = "termination"
        else:
            reward = 0.0
            curr_node = random.choice(list(filter(lambda tp: tp[2]["type"] == action, G.edges(curr_node, data=True))))[0]
            state = dec(emb(torch.tensor([action_to_ix[action]], dtype=torch.long).to(device)), state)
