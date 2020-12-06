import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math, copy, time
import pandas as pd
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils import *
from IPython.core.debugger import set_trace

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# we will use CUDA if it is available
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda:0') if USE_CUDA else torch.device("cpu")

# set random seed
seed = 666
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, trg_embed, q_function):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.trg_embed = trg_embed
        self.q_function = q_function

    def forward(self, src, trg, src_mask, trg_mask, src_lengths, trg_lengths):
        """Take in and process masked src and target sequences."""
        encoder_hidden, encoder_final = self.encode(src, src_mask, src_lengths)
        return self.decode(encoder_hidden, encoder_final, src_mask, trg, trg_mask)

    def encode(self, src, src_mask, src_lengths):
        return self.encoder(self.src_embed(src), src_mask, src_lengths)

    def decode(self, encoder_hidden, encoder_final, src_mask, trg, trg_mask, decoder_hidden=None):
        return self.decoder(self.trg_embed(trg), encoder_hidden, encoder_final, src_mask, trg_mask, hidden=decoder_hidden)


class Encoder(nn.Module):
    """Encodes a sequence of word embeddings"""

    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=dropout)

    def forward(self, x, mask, lengths):
        """
        Applies a bidirectional GRU to sequence of embeddings x.
        The input mini-batch x needs to be sorted by length.
        x should have dimensions [batch, time, dim].
        """
        packed = pack_padded_sequence(x, lengths, batch_first=True)
        output, final = self.rnn(packed)
        output, _ = pad_packed_sequence(output, batch_first=True)

        # we need to manually concatenate the final states for both directions
        fwd_final = final[0:final.size(0):2]
        bwd_final = final[1:final.size(0):2]
        final = torch.cat([fwd_final, bwd_final], dim=2)  # [num_layers, batch, 2*dim]

        return output, final


class Decoder(nn.Module):
    """A conditional RNN decoder with attention."""

    def __init__(self, emb_size, hidden_size, attention, num_layers=1, dropout=0.0, bridge=True):
        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.attention = attention
        self.dropout = dropout

        self.rnn = nn.GRU(emb_size + 2 * hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout)

        # to initialize from the final encoder state
        self.bridge = nn.Linear(2 * hidden_size, hidden_size, bias=True) if bridge else None

        self.dropout_layer = nn.Dropout(p=dropout)
        self.pre_output_layer = nn.Linear(hidden_size + 2*hidden_size + emb_size, hidden_size, bias=False)

    def forward_step(self, prev_embed, encoder_hidden, src_mask, proj_key, hidden):
        """Perform a single decoder step (1 word)"""

        # compute context vector using attention mechanism
        query = hidden[-1].unsqueeze(1)  # [#layers, B, D] -> [B, 1, D]
        context, attn_probs = self.attention(query=query, proj_key=proj_key, value=encoder_hidden, mask=src_mask)

        # update rnn hidden state
        rnn_input = torch.cat([prev_embed, context], dim=2)
        output, hidden = self.rnn(rnn_input, hidden)

        pre_output = torch.cat([prev_embed, output, context], dim=2)
        pre_output = self.dropout_layer(pre_output)
        pre_output = self.pre_output_layer(pre_output)

        return output, hidden, pre_output

    def forward(self, trg_embed, encoder_hidden, encoder_final, src_mask, trg_mask, hidden=None, max_len=None):
        """Unroll the decoder one step at a time."""

        # the maximum number of steps to unroll the RNN
        if max_len is None:
            max_len = trg_mask.size(-1)

        # initialize decoder hidden state
        if hidden is None:
            hidden = self.init_hidden(encoder_final)

        # pre-compute projected encoder hidden states
        # (the "keys" for the attention mechanism)
        # this is only done for efficiency
        proj_key = self.attention.key_layer(encoder_hidden)

        # here we store all intermediate hidden states and pre-output vectors
        decoder_states = []
        pre_output_vectors = []

        # unroll the decoder RNN for max_len steps
        for i in range(max_len):
            prev_embed = trg_embed[:, i].unsqueeze(1)
            output, hidden, pre_output = self.forward_step(prev_embed, encoder_hidden, src_mask, proj_key, hidden)
            decoder_states.append(output)
            pre_output_vectors.append(pre_output)

        decoder_states = torch.cat(decoder_states, dim=1)
        pre_output_vectors = torch.cat(pre_output_vectors, dim=1)
        return decoder_states, hidden, pre_output_vectors  # [B, N, D]

    def init_hidden(self, encoder_final):
        """Returns the initial decoder state,
        conditioned on the final encoder state."""

        if encoder_final is None:
            return None  # start with zeros

        return torch.tanh(self.bridge(encoder_final))


class BahdanauAttention(nn.Module):
    """Implements Bahdanau (MLP) attention"""

    def __init__(self, hidden_size, key_size=None, query_size=None):
        super(BahdanauAttention, self).__init__()

        # We assume a bi-directional encoder so key_size is 2*hidden_size
        key_size = 2*hidden_size if key_size is None else key_size
        query_size = hidden_size if query_size is None else query_size

        self.key_layer = nn.Linear(key_size, hidden_size, bias=False)
        self.query_layer = nn.Linear(query_size, hidden_size, bias=False)
        self.energy_layer = nn.Linear(hidden_size, 1, bias=False)

        # to store attention scores
        self.alphas = None

    def forward(self, query=None, proj_key=None, value=None, mask=None):
        assert mask is not None, "mask is required"

        # We first project the query (the decoder state).
        # The projected keys (the encoder states) were already pre-computated.
        query = self.query_layer(query)

        # Calculate scores.
        scores = self.energy_layer(torch.tanh(query + proj_key))
        scores = scores.squeeze(2).unsqueeze(1)

        # Mask out invalid positions.
        # The mask marks valid positions so we invert it using `mask & 0`.
        scores.data.masked_fill_(mask == 0, -float('inf'))

        # Turn scores to probabilities.
        alphas = F.softmax(scores, dim=-1)
        self.alphas = alphas

        # The context vector is the weighted sum of the values.
        context = torch.bmm(alphas, value)

        # context shape: [B, 1, 2D], alphas shape: [B, 1, M]
        return context, alphas


class QFunction(nn.Module):
    """Define standard linear action value function."""
    def __init__(self, hidden_size, vocab_size):
        super(QFunction, self).__init__()
        self.proj = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, x):
        return self.proj(x)


def make_model(src_vocab, tgt_vocab, emb_size=256, hidden_size=512, num_layers=1, dropout=0.1):
    "Helper: Construct a model from hyperparameters."

    attention = BahdanauAttention(hidden_size)

    model = EncoderDecoder(
        Encoder(emb_size, hidden_size, num_layers=num_layers, dropout=dropout),
        Decoder(emb_size, hidden_size, attention, num_layers=num_layers, dropout=dropout),
        nn.Embedding(src_vocab, emb_size),
        nn.Embedding(tgt_vocab, emb_size),
        QFunction(hidden_size, tgt_vocab))

    return model


class Batch:
    """Object for holding a batch of data with mask during training.
    Input is a batch from a torch text iterator.
    """

    def __init__(self, src, trg, pad_index=0):

        src, src_lengths = src

        self.src = src
        self.src_lengths = src_lengths
        self.src_mask = (src != pad_index).unsqueeze(-2)
        self.nseqs = src.size(0)

        trg, trg_lengths = trg

        self.trg = trg
        self.trg_lengths = trg_lengths
        self.trg_mask = (self.trg != pad_index)
        self.ntokens = self.trg_mask.data.sum().item()


class SimpleLossCompute:
    """A simple loss compute and train function."""

    def __init__(self, q_function, criterion, opt=None):
        self.q_function = q_function
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.q_function(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1))
        loss = loss / norm

        if self.opt is not None:
            loss.backward()
            self.opt.step()
            self.opt.zero_grad()

        return loss.data.item() * norm


def simulate_episode(G, qa_instance, model, action_to_ix, max_len, epsilon):

    question, decorated_entity, answer_set = qa_instance
    tokenized_inputs = tokenizer(question, max_length=50, padding=True, truncation=True, return_tensors="pt")
    src, src_mask = tokenized_inputs["input_ids"].to(DEVICE), tokenized_inputs["attention_mask"].unsqueeze(-2).to(DEVICE)
    assert decorated_entity in G.nodes
    kgnode = decorated_entity
    print(question)
    print(kgnode)

    kgnode_chain = []
    action_chain = []
    reward_chain = []

    encoder_hidden, encoder_final = model.encode(src, src_mask, [src_mask.sum().item()])

    # pre-compute projected encoder hidden states
    # (the "keys" for the attention mechanism)
    # this is only done for efficiency
    proj_key = model.decoder.attention.key_layer(encoder_hidden)

    # initialize decoder hidden state
    init_hidden = model.decoder.init_hidden(encoder_final)
    init_embed = model.trg_embed(torch.tensor([action_to_ix["[SOS]"]], device=DEVICE)).unsqueeze(1)
    _, context, _ = model.decoder.forward_step(init_embed, encoder_hidden, src_mask, proj_key, init_hidden)

    for t in range(max_len):
        # compute the action value functions for available actions at the current node
        actions = unique([info["type"] for (_, _, info) in G.edges(kgnode, data=True)]) + ["terminate"]
        values = model.q_function(context)[0, 0, [action_to_ix[action] for action in actions]]

        print(actions)
        print(values.data.reshape(-1).to("cpu"))

        # select the action at the current time step with epsilon-greedy policy
        if random.random() < epsilon:
            action = random.choice(actions)
        else:
            action = actions[values.argmax()]

        # take the action
        if (action == "terminate") or (t == max_len-1):
            reward = torch.tensor(1.0 if (re.match(r".+: (.+)", kgnode).group(1) in answer_set) else 0.0).to(DEVICE)
            kgnode_next = "termination"
            context_next = None
            print(action, "  =====>  ", kgnode_next)
        else:
            reward = torch.tensor(0.0).to(DEVICE)
            kgnode_next = random.choice(list(filter(lambda tp: tp[2]["type"] == action, G.edges(kgnode, data=True))))[1]
            action_embed = model.trg_embed(torch.tensor([action_to_ix[action]], device=DEVICE)).unsqueeze(1)
            _, context_next, _ = model.decoder.forward_step(action_embed, encoder_hidden, src_mask, proj_key, context)
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


def make_batch(episodes, tokenizer, action_to_ix, pad_index=0, sos_index=1):
    episodes = sorted(episodes, key=lambda x: (-len(tokenizer.tokenize(x.qa_instance.question)), -len(x.action_chain)))

    inputs = tokenizer(list(map(lambda x: x.qa_instance.question, episodes)), max_length=50, padding=True, truncation=True, return_tensors="pt", return_length=True)
    src = inputs["input_ids"].to(DEVICE)
    src_lengths = inputs["length"]

    max_len = max(len(x.action_chain) for x in episodes)
    trg = torch.cat(tuple(map(lambda x: torch.tensor([[sos_index] + [action_to_ix[action] for action in x.action_chain] + [pad_index]*(max_len-len(x.action_chain))], device=DEVICE), episodes)), dim=0)
    trg_lengths = list(map(lambda x: len(x.action_chain)+1, episodes))

    kgnode_chains = [episode.kgnode_chain for episode in episodes]
    action_chains = [episode.action_chain for episode in episodes]
    reward_chains = [episode.reward_chain for episode in episodes]

    return Batch((src, src_lengths), (trg, trg_lengths), pad_index=pad_index), kgnode_chains, action_chains, reward_chains


def compute_loss(episodes, tokenizer, model, action_to_ax):

    batch, kgnode_chains, action_chains, reward_chains = make_batch(episodes, tokenizer, action_to_ix)
    _, _, pre_output = model.forward(batch.src, batch.trg, batch.src_mask, batch.trg_mask, batch.src_lengths, batch.trg_lengths)

    batch_values = model.q_function(pre_output)

    losses = []
    for (i, (kgnode_chain, action_chain, reward_chain)) in enumerate(zip(kgnode_chains, action_chains, reward_chains)):
        for t in range(len(kgnode_chain)):
            kgnode, action, reward = kgnode_chain[t], action_chain[t], reward_chain[t]

            if t != len(kgnode_chain)-1:
                kgnode_next = kgnode_chain[t+1]
                actions_next = unique([info["type"] for (_, _, info) in G.edges(kgnode_next, data=True)]) + ["terminate"]
                values_next = batch_values[i, t+1, [action_to_ix[action] for action in actions_next]]
                reference = reward + gamma*values_next.max().item()
            else:
                reference = reward

            losses.append(loss_func(batch_values[i, t, action_to_ix[action]], reference))
            print("    ", kgnode, "    ", action, "    ", batch_values[i, t, action_to_ix[action]].data.to("cpu").item(), "    ", reference.to("cpu").item())

    return sum(losses)


if __name__ == "__main__":

    max_len = 3
    gamma = 0.90
    epsilon_start = 1.00
    epsilon_end = 0.10
    decay_rate = 5.00
    M = 10000
    batch_size = 1

    entity_token = "[ETY]"
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", additional_special_tokens=[entity_token])

    G = read_MetaQA_KG()

    qa_train_1h, qa_dev_1h, qa_test_1h = read_MetaQA_Instances("1-hop", entity_token, DEVICE)
#   qa_train_2h, qa_dev_2h, qa_test_2h = read_MetaQA_Instances("2-hop", entity_token, DEVICE)
#   qa_train_3h, qa_dev_3h, qa_test_3h = read_MetaQA_Instances("3-hop", entity_token, DEVICE)

    qa_train = pd.concat([qa_train_1h])
    qa_dev = pd.concat([qa_dev_1h])
    qa_test = pd.concat([qa_test_1h])

#    qa_train = pd.concat([qa_train_1h, qa_train_2h, qa_train_3h])
#    qa_dev = pd.concat([qa_dev_1h, qa_dev_2h, qa_dev_3h])
#    qa_test = pd.concat([qa_test_1h, qa_test_2h, qa_test_3h])

    possible_actions = ["[PAD]", "[SOS]"] + sorted(list(set([edge[2]["type"] for edge in G.edges(data=True)]))) + ["terminate"]
    action_to_ix = dict(map(reversed, enumerate(possible_actions)))

    model = make_model(len(tokenizer), len(possible_actions), emb_size=32, hidden_size=64, dropout=0.0).to(DEVICE)
    loss_func = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), betas=(0.9, 0.999), weight_decay=2.5e-4)

    memory_overall = ReplayMemory(1000)
    memory_success = ReplayMemory(1000)
    success_rate = 0.0
    for m in range(M):
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * math.exp(-decay_rate * (m / M))
        print("epsilon: {:5.3f}".format(epsilon))

        qa_instance = qa_train.sample(1).values[0]

        with torch.no_grad():
            qa_instance, kgnode_chain, action_chain, reward_chain = simulate_episode(G, qa_instance, model, action_to_ix, max_len, 0.90)
        print("success" if reward_chain[-1] == 1.0 else "failure")
        print()

        success_rate = 0.999 * success_rate + 0.001 * reward_chain[-1]
        print("success_rate: {:5.3f}".format(success_rate))
        print()

        memory_overall.push(Episode(qa_instance, kgnode_chain, action_chain, reward_chain))
        if reward_chain[-1] == 1.0:
            memory_success.push(Episode(qa_instance, kgnode_chain, action_chain, reward_chain))

        episodes = memory_overall.sample_last(batch_size)
        for t in range(max_len):
            # optimize model
            loss = compute_loss(episodes, tokenizer, model, action_to_ix)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print()

#       if (len(memory_success) > 0):
#           succeed_episodes = memory_success.sample_random(batch_size)
#           for t in range(T):
#               loss = compute_loss(episodes, tokenizer, model, action_to_ix)
#               optimizer.zero_grad()
#               loss.backward()
#               optimizer.step()
#               print()

        print()
        print(flush=True)
