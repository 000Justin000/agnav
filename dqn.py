import re
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoModel
from utils import *

class ActionValueFunc(nn.Module):
    def __init__(self, ndim_context, ndim_action):
        super(ActionValueFunc, self).__init__()
        self.context_transformation = nn.Sequential(nn.Linear(ndim_context,ndim_action), nn.Tanh())

    def forward(self, context, emb_actions):
        transformed_context = self.context_transformation(context)
        vals = F.softplus(F.cosine_similarity(transformed_context, emb_actions), beta=10)
        return vals


torch.autograd.set_detect_anomaly(True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# the main function starts here

# 1. the true MDP state is (question, all_actions_upto_now, all_kgnodes_upto_now) 
#    we encode that with (current_kgnode, current_context); furthermore, we treat 
#    nodes as abstract objects (part of the knowledge graph) that does not have 
#    any meaning on their own, therefore, we exclude them in computing the context
#    transition or the action-value function

#---------------------------------------------------------------------
# set hyper-parameters
#---------------------------------------------------------------------
ndim_context = 768
ndim_action = 768
T = 3
gamma = 0.90
epsilon_start = 1.00
epsilon_end = 0.10
decay_rate = 5.00
M = 100000
#---------------------------------------------------------------------

#---------------------------------------------------------------------
# get the knowledge graph, question/answers instances
#---------------------------------------------------------------------
G = read_MetaQA_KG()
qa_train, qa_dev, qa_test = read_MetaQA_Instances("1-hop", device)
#---------------------------------------------------------------------

#---------------------------------------------------------------------
# determine all possible actions
#---------------------------------------------------------------------
possible_actions = sorted(list(set([edge[2]["type"] for edge in G.edges(data=True)])))
possible_actions.append("terminate")
action_to_ix = dict(map(reversed, enumerate(possible_actions)))
#---------------------------------------------------------------------

#---------------------------------------------------------------------
# trainable models
#---------------------------------------------------------------------
# action_embeddings
emb = nn.Embedding(len(possible_actions), ndim_action).to(device)
# language embedding
enc = AutoModel.from_pretrained("bert-base-uncased").to(device)
# action value function
qsa = ActionValueFunc(ndim_context, ndim_action).to(device)
# context transation function
dec = nn.GRUCell(ndim_action, ndim_context).to(device)
#---------------------------------------------------------------------

#---------------------------------------------------------------------
# optimizer setup
#---------------------------------------------------------------------
optimizer = optim.Adam([{"params": emb.parameters(), "lr": 1.0e-4},
                        {"params": enc.parameters(), "lr": 0.0e-4},
                        {"params": qsa.parameters(), "lr": 1.0e-4},
                        {"params": dec.parameters(), "lr": 1.0e-4}])
loss_func = nn.MSELoss()
#---------------------------------------------------------------------


def emb_actions(actions):
    return emb(torch.tensor([action_to_ix[action] for action in actions], dtype=torch.long).to(device))


def simulate_episode(epsilon):

    qa_instance = qa_train.sample(1).values[0]
    # qa_instance = qa_train[[0,12570,22542,34419,35610,51705,75016]].sample(1).values[0]
    # qa_instance = qa_train[[12570,22542]].sample(1).values[0]
    question, tokenized_inputs, decorated_entity, answer_set = qa_instance
    assert decorated_entity in G.nodes

    kgnode_chain = []
    action_chain = []
    reward_chain = []

    # set initial values for context and action
    kgnode = decorated_entity
    context = enc(**tokenized_inputs)[1]

    print(question)
    print(kgnode)

    for t in range(T):
        # compute the action value functions for available actions at the current node
        actions = unique([info["type"] for (_, _, info) in G.edges(kgnode, data=True)]) + ["terminate"]
        values = qsa(context, emb_actions(actions))

        print(actions)
        print(values.data.reshape(-1).to("cpu"))

        # select the action at the current time step with epsilon-greedy policy
        if random.random() < epsilon:
            action = random.choice(actions)
        else:
            action = actions[values.argmax()]

        # take the action
        if (action == "terminate") or (t == T-1):
            reward = torch.tensor(1.0 if (re.match(r".+: (.+)", kgnode).group(1) in answer_set) else 0.0).to(device)
            kgnode_next = "termination"
            context_next = None
            print(action, "  =====>  ", kgnode_next)
        else:
            reward = torch.tensor(0.0).to(device)
            kgnode_next = random.choice(list(filter(lambda tp: tp[2]["type"] == action, G.edges(kgnode, data=True))))[1]
            context_next = dec(emb_actions([action]), context)
            print(action, "  =====>  ", kgnode_next)

        kgnode_chain.append(kgnode)
        action_chain.append(action)
        reward_chain.append(reward)

        if kgnode_next == "termination":
            break
        else:
            kgnode = kgnode_next
            context = context_next

    return qa_instance, kgnode_chain, action_chain, reward_chain


def replay_episode(episode):

    qa_instance, kgnode_chain, action_chain, reward_chain = episode
    question, tokenized_inputs, decorated_entity, answer_set = qa_instance

    print(question)
    context = enc(**tokenized_inputs)[1]

    losses = []
    for t in range(len(kgnode_chain)):
        kgnode, action, reward = kgnode_chain[t], action_chain[t], reward_chain[t]

        if t != len(kgnode_chain)-1:
            kgnode_next = kgnode_chain[t+1]
            context_next = dec(emb_actions([action]), context)
            actions_next = unique([info["type"] for (_, _, info) in G.edges(kgnode_next, data=True)]) + ["terminate"]
            values_next = qsa(context_next, emb_actions(actions_next))
            reference = reward + gamma*values_next.max().item()
        else:
            reference = reward

        losses.append(loss_func(qsa(context, emb_actions([action])), reference))
        print("    ", kgnode, "    ", action, "    ", qsa(context, emb_actions([action]))[0].data.to("cpu"), "    ", reference.to("cpu"))

        if t != len(kgnode_chain)-1:
            context = context_next

    return sum(losses)

memory_overall = ReplayMemory(1000)
memory_success = ReplayMemory(1000)
success_rate = 0.0
for m in range(M):
    epsilon = epsilon_end + (epsilon_start-epsilon_end) * math.exp(-decay_rate*(m/M))
    print("epsilon: {:5.3f}".format(epsilon))

    with torch.no_grad():
        qa_instance, kgnode_chain, action_chain, reward_chain = simulate_episode(epsilon)
    print("success" if reward_chain[-1] == 1.0 else "failure")
    print()

    success_rate = 0.999*success_rate + 0.001*reward_chain[-1]
    print("success_rate: {:5.3f}".format(success_rate))
    print()

    memory_overall.push(Episode(qa_instance, kgnode_chain, action_chain, reward_chain))
    if reward_chain[-1] == 1.0:
        memory_success.push(Episode(qa_instance, kgnode_chain, action_chain, reward_chain))

    last_episode = memory_overall.sample_last(1)[0]
    for t in range(T):
        loss = replay_episode(last_episode)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print()

    if (len(memory_success) > 0) and (random.random() < 0.3):
        succeed_episode = memory_success.sample_random(1)[0]
        for t in range(T):
            loss = replay_episode(succeed_episode)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print()

    print(flush=True)

    if (m+1) % 10000 == 0:
        torch.save({"emb" : emb.state_dict(), "enc" : enc.state_dict(), "qsa" : qsa.state_dict(), "dec" : dec.state_dict()}, "checkpoints/save@{:07d}.pt".format(m+1))
