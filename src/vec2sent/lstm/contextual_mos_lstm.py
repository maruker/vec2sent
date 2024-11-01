from typing import List, Tuple, Dict
from typing_extensions import TypedDict

import torch
import torch.nn as nn
from torch.autograd import Variable
from huggingface_hub import PyTorchModelHubMixin

from vec2sent.mos import embedded_dropout, WeightDrop, RNNModel

ModelConfig = TypedDict("ModelConfig", {
    "ntoken": int,
    "ninp": int,
    "context_size": int,
    "nhid": int,
    "nhidlast": int,
    "nlayers": int,
    "dropout": float,
    "dropouth": float,
    "dropouti": float,
    "dropoute": float,
    "wdrop": float,
    "tie_weights": bool,
    "ldropout": float,
    "n_experts": int,
    "options": Dict
})


class ConditionedRNNModel(RNNModel, PyTorchModelHubMixin):

    def __init__(self, config: ModelConfig):
        """
        This class modifies the Mixture of Softmaxes model from
        "Breaking the Softmax Bottleneck: A High-Rank Language Model"
        by adding a context vector as an input.
        Container module with an encoder, a recurrent module, and a decoder.

        @param config.ntoken: Number of tokens in the vocabulary
        @param config.ninp: Size of each embedding vector
        @param config.context_size: Size of the context vector
        @param config.nhid: Size of the hidden state
        @param config.nhidlast: Size of the last hidden state or output layer
        @param config.nlayers: Number of hidden layers (including input and output)
        @param config.dropout: Dropout probability.
        @param config.dropouth: Dropout probability for hidden layers
        @param config.dropouti: Dropout probability for input layer
        @param config.dropoute: Dropout probability for embedding layer
        @param config.wdrop: Weight dropout probability
        @param config.tie_weights: Reuse (and train together) embeddings and output layer
        @param config.ldropout: Dropout probability for latent layer
        @param config.n_experts: Number of experts/ Number of softmax calculations
        @param config.options: Options that were passed to the sentence embedding during training
        """
        self.device = None
        super_args = config.copy()
        super_args.pop("context_size")
        super_args.pop("options", None)
        super().__init__("LSTM", **super_args)
        self.config = config  # Make sure PyTorchModelHubMixin can save the constructor arguments

        self.context_size = config["context_size"]

        rnns = [
            torch.nn.LSTM(config["ninp"] + config["context_size"] if layer == 0 else config["nhid"],
                          config["nhid"] if layer != config["nlayers"] - 1 else config["nhidlast"], 1, dropout=0)
            for layer in range(config["nlayers"])]
        if config["wdrop"]:
            rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=config["wdrop"] if self.use_dropout else 0) for rnn in
                    rnns]
        self.rnns = torch.nn.ModuleList(rnns)

        self.lockdrop = LockedDropout1d()

    def to(self, device: torch.device) -> "ConditionedRNNModel":
        super().to(device)
        self.device = device
        return self

    def forward(self, inputs, contexts, hidden=None, return_h=False, return_prob=False):
        if hidden is None:
            hidden = self.init_hidden(len(inputs))

        if not isinstance(contexts, list):
            # If we only get one context
            contexts = [contexts]

        embs = []
        for i, input in enumerate(inputs):
            if self.device is not None:
                input = input.to(self.device)
            emb = embedded_dropout(self.encoder, input,
                                   dropout=self.dropoute if (self.training and self.use_dropout) else 0)
            context = contexts[i]
            context = torch.stack([context] * emb.size(0))
            if self.device is not None:
                context = context.to(self.device)
            emb = torch.cat([emb, context], 1)
            emb = self.lockdrop(emb, self.dropouti if self.use_dropout else 0)
            embs.append(emb)

        emb = nn.utils.rnn.pack_sequence(embs, enforce_sorted=False)

        raw_output = emb
        new_hidden: List[Tuple[torch.Tensor, torch.Tensor]] = []
        # raw_output, hidden = self.rnn(emb, hidden)
        raw_outputs = []
        outputs = []
        for l, rnn in enumerate(self.rnns):
            raw_output, new_h = rnn(raw_output, hidden[l])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if l != self.nlayers - 1:
                # self.hdrop(raw_output)
                raw_output = self.lockdrop(raw_output, self.dropouth if self.use_dropout else 0)
                outputs.append(raw_output)
        hidden = new_hidden

        raw_output, lengths = torch.nn.utils.rnn.pad_packed_sequence(raw_output)

        output = self.lockdrop(raw_output, self.dropout if self.use_dropout else 0)
        outputs.append(output)

        latent = self.latent(output)
        latent = self.lockdrop(latent, self.dropoutl if self.use_dropout else 0)
        logit = self.decoder(latent.view(-1, self.ninp))

        prior_logit = self.prior(output).contiguous().view(-1, self.n_experts)
        prior = nn.functional.softmax(prior_logit, -1)

        prob = nn.functional.softmax(logit.view(-1, self.n_experts, self.ntoken).view(-1, self.ntoken), -1).view(-1,
                                                                                                                 self.n_experts,
                                                                                                                 self.ntoken)
        prob = (prob * prior.unsqueeze(2).expand_as(prob)).sum(1)
        prob = prob.view(prob.size(0), 1, prob.size(-1))

        if return_prob:
            model_output = prob
        else:
            log_prob = torch.log(prob + 1e-8)
            model_output = log_prob

        if return_h:
            return model_output, hidden, raw_outputs, outputs
        return model_output, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return [(Variable(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else self.nhidlast).zero_()),
                 Variable(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else self.nhidlast).zero_()))
                for l in range(self.nlayers)]

    def init_embeddings(self, embeddings):
        self.encoder = embeddings
        self.encoder.weight.requires_grad = False
        self.decoder.weight = self.encoder.weight


class LockedDropout1d(nn.Module):
    """
    Original MoS implementation of locked dropout was modified to deal with packed sequences
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, dropout=0.5):
        if not self.training or not dropout:
            return x

        # Deal with padded and packed sequences
        return_packed_sequence = False
        if isinstance(x, torch.nn.utils.rnn.PackedSequence):
            x, lengths = torch.nn.utils.rnn.pad_packed_sequence(x)
            return_packed_sequence = True

        # Deal with 1d and 2d inputs
        if len(x.size()) > 2:
            size = (1,) + x.shape[1:]
        else:
            size = x.size()

        m = x.data.new(size=size).bernoulli_(1 - dropout)
        # mask = Variable(m, requires_grad=False) / (1 - dropout)
        mask = Variable(m.div_(1 - dropout), requires_grad=False)
        mask = mask.expand_as(x)
        result = mask * x

        if return_packed_sequence:
            return torch.nn.utils.rnn.pack_padded_sequence(result, lengths, enforce_sorted=False)
        return x
