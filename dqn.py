import re
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModel
from utils import read_MetaQA_KG, read_MetaQA_Instances

class ActionValueFunc(nn.Module):
    def __init__(self, ndim_state, ndim_action):
        super(ActionValueFunc, self).__init__()
        self.state_transformation = nn.Sequential(nn.Linear(ndim_state,ndim_action))

    def forward(self, state, emb_actions):
        transformed_state = self.state_transformation(state)
        # vals = torch.sigmoid((transformed_state*emb_actions).sum(axis=-1))
        vals = (transformed_state*emb_actions).sum(axis=-1)
        return vals


torch.autograd.set_detect_anomaly(True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ndim_state = 768
ndim_action = 128
gamma = 0.90
epsilon_start = 1.00
epsilon_end = 0.10
M = 10000
epsilon_decay = 0.2*M
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
qsa = ActionValueFunc(ndim_state, ndim_action).to(device)
# state transation function
dec = nn.RNNCell(ndim_action, ndim_state).to(device)
#---------------------------------------------------------------------

emb_actions = lambda actions: emb(torch.tensor([action_to_ix[action] for action in actions], dtype=torch.long).to(device))
unique = lambda items: list(set(items))

optimizer = optim.Adam([{"params": emb.parameters(), "lr": 1.0e-4},
                        {"params": enc.parameters(), "lr": 1.0e-4},
                        {"params": qsa.parameters(), "lr": 1.0e-4},
                        {"params": dec.parameters(), "lr": 1.0e-4}])
loss_func = nn.L1Loss()

success_rate = 0.0
for m in range(M):
    epsilon = epsilon_end + (epsilon_start-epsilon_end) * math.exp(-m/epsilon_decay)

    question, tokenized_inputs, decorated_entity, answer_set = qa_train.sample(1).values[0]
    # question, tokenized_inputs, decorated_entity, answer_set = qa_train[0]
    assert decorated_entity in G.nodes

    # set initial values for state, action, and reward
    curr_node = decorated_entity
    curr_state = enc(**tokenized_inputs)[1]
    curr_actions = None
    curr_values = None

    print(question)
    print(curr_node)

    losses = []
    for t in range(T):
        # compute the action value functions for available actions at the current node
        if t == 0:
            curr_actions = unique([info["type"] for (_, _, info) in G.edges(curr_node, data=True)]) + ["terminate"]
            curr_values = qsa(curr_state, emb_actions(curr_actions))
        else:
            # values already computed
            pass

        print(curr_actions)
        print(curr_values.data.reshape(-1).to("cpu"))

        # select the action at the current time step
        if t == T-1:
            action = "terminate"
        else:
            # selection action with epsilon-greedy policy
            if random.random() < epsilon:
                action = random.choice(curr_actions)
            else:
                action = curr_actions[curr_values.argmax()]

        # take the action
        if action != "terminate":
            reward = torch.tensor(0.0, device=device)
            next_node = random.choice(list(filter(lambda tp: tp[2]["type"] == action, G.edges(curr_node, data=True))))[1]
            next_state = dec(emb_actions([action]), curr_state)
            print(action, "  =====>  ", next_node)
        else:
            reward = torch.tensor(1.0 if (re.match(r".+: (.+)", curr_node).group(1) in answer_set) else 0.0, device=device)
            next_node = "termination"
            next_state = None
            print(action, "  =====>  ", next_node)

        # temper difference error as loss of this step
        if next_node != "termination":
            next_actions = unique([info["type"] for (_, _, info) in G.edges(next_node, data=True)]) + ["terminate"]
            next_values = qsa(next_state, emb_actions(next_actions))
            reference = reward + gamma * next_values.max().item()
        else:
            next_actions = None
            next_values = None
            reference = reward
        losses.append(loss_func(qsa(curr_state, emb_actions([action])), reference))
        # losses.append(abs(qsa(curr_state, emb_actions([action]) - reference)))
        print(curr_node, "    ", action, "    ", qsa(curr_state, emb_actions([action]))[0].data.to("cpu"), "    ", reference.to("cpu"))

        if next_node != "termination":
            curr_node = next_node
            curr_state = next_state
            curr_actions = next_actions
            curr_values = next_values
        else:
            success_rate = 0.999*success_rate + 0.001*reward
            print("success" if reward == 1.0 else "failure")
            print("success_rate:    ", float(success_rate))
            break

    optimizer.zero_grad()
    sum(losses).backward()
    optimizer.step()
    print()

    if (m+1) % 10000 == 0:
        torch.save({"emb" : emb.state_dict(), "enc" : enc.state_dict(), "qsa" : qsa.state_dict(), "dec" : dec.state_dict()}, "checkpoints/save@{:07d}.pt".format(m+1))
