import re
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModel
from utils import read_MetaQA_KG, read_MetaQA_Instances

torch.autograd.set_detect_anomaly(True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ndim_state = 768
ndim_action = 128
gamma = 0.90
epsilon_start = 1.00
epsilon_end = 0.10
M = 1000000
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
qsa = nn.Sequential(nn.Linear(ndim_action+ndim_state,ndim_action+ndim_state), nn.Sigmoid(), nn.Linear(ndim_action+ndim_state,1), nn.Sigmoid()).to(device)
# state transation function
dec = nn.RNNCell(ndim_action, ndim_state).to(device)
#---------------------------------------------------------------------

optimizer = optim.Adam([{"params": emb.parameters(), "lr": 1.0e-4},
                        {"params": enc.parameters(), "lr": 1.0e-4},
                        {"params": qsa.parameters(), "lr": 1.0e-4},
                        {"params": dec.parameters(), "lr": 1.0e-4}])
success_rate = 0.0
for m in range(M):
    epsilon = epsilon_end + (epsilon_start-epsilon_end) * math.exp(-m/epsilon_decay)

    question, tokenized_inputs, decorated_entity, answer_set = qa_train.sample(1).values[0]
    print(question)
    assert decorated_entity in G.nodes
    curr_node = decorated_entity
    print(curr_node)

    # set initial values for state, action, and reward
    state = enc(**tokenized_inputs)[1]
    action = None
    reward = None

    losses = []
    for t in range(T+1):
        if curr_node != "termination":
            # available actions as going along one of the edges or terminate
            actions = list(set([info["type"] for (curr_node, next_node, info) in G.edges(curr_node, data=True)] + ["terminate"]))
            emb_actions = emb(torch.tensor([action_to_ix[action] for action in actions], dtype=torch.long).to(device))
            emb_sapairs = torch.cat((state.repeat(len(actions), 1), emb_actions), 1)
            val_sapairs = qsa(emb_sapairs)

        if t != 0:
            reference = reward if (curr_node == "termination") else (reward + gamma*val_sapairs.max().item())
            val_sapair0 = qsa(torch.cat((state, emb(torch.tensor([action_to_ix[action]], dtype=torch.long).to(device))), 1))
            # store the loss at each time step (to be used for optimization at the end of the episode)
            losses.append(nn.functional.smooth_l1_loss(val_sapair0, reference))

        if curr_node == "termination":
            break
        else:
            if t == T-1:
                action = "terminate"
            else:
                # selection action with epsilon-greedy policy
                if random.random() < epsilon:
                    action = random.choice(actions)
                else:
                    action = actions[val_sapairs.argmax()]

            # take the action
            if action != "terminate":
                reward = torch.tensor(0.0, device=device)
                curr_node = random.choice(list(filter(lambda tp: tp[2]["type"] == action, G.edges(curr_node, data=True))))[1]
                state = dec(emb(torch.tensor([action_to_ix[action]], dtype=torch.long).to(device)), state)
                print(action, "  =====>  ", curr_node)
            else:
                reward = torch.tensor(1.0 if (re.match(r".+: (.+)", curr_node).group(1) in answer_set) else 0.0, device=device)
                curr_node = "termination"
                success_rate = 0.999*success_rate + 0.001*reward
                print(action, "  =====>  ", curr_node)
                print("success" if reward == 1.0 else "failure")
                print("success_rate:    ", float(success_rate))
                print()

    optimizer.zero_grad()
    sum(losses).backward()
    optimizer.step()

    if (m+1) % 10000 == 0:
        torch.save({"emb" : emb.state_dict(), "enc" : enc.state_dict(), "qsa" : qsa.state_dict(), "dec" : dec.state_dict()}, "checkpoints/save@{:07d}.pt".format(m+1))
