from typing import List, Callable, Tuple

import torch
from queue import PriorityQueue


class BeamSearchNode:
    def __init__(self, hidden, previous_node, word_id, log_prob, length):
        self.hidden = hidden
        self.prev = previous_node
        self.log_prob = log_prob
        self.len = length
        self.word_id = word_id

    def eval(self):
        return self.log_prob / self.length_penalty(1)

    def length_penalty(self, alpha):
        return (self.len + 5 / 6) ** alpha

    def __lt__(self, other):
        # Return true if self has higher probability than other
        return self.eval() > other.eval()


def beam_decode(
        decoder: Callable[[List[int], List[Tuple[torch.Tensor, torch.Tensor]]],
        Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]],
        init_hidden: torch.FloatTensor,
        sos: int,
        eos: int,
        max_len: int,
        beam_width: int = 10,
        top_k: int = 1
) -> List[List[int]]:
    """
    @param decoder: lambda word_idx (int) -> probabilities of next word (FloatTensor)
    @param init_hidden: initial hidden state (FloatTensor)
    @param sos: start of sentence (word index)
    @param eos: Index of end of sentence token
    @param max_len: Maximum length of outputs
    @param beam_width: Number of words generated at each step
    @param top_k: Number of generated sentences

    @return: List of generated sentences where each sentence is a list of word indices.
    """
    n_required = 5

    nodes: PriorityQueue[BeamSearchNode] = PriorityQueue()
    next_nodes = PriorityQueue()
    results = []

    def decode_beams():
        batch_size = min(nodes.qsize(), beam_width)

        batch = []
        while len(batch) < batch_size:
            if nodes.qsize() == 0:
                break
            n = nodes.get(False)
            if n.word_id == eos:
                results.append(n)
            elif n.len <= max_len:
                batch.append(n)

        if len(batch) == 0:
            return

        batch_size = len(batch)

        inputs = [n.word_id for n in batch]
        prev_hidden = [n.hidden for n in batch]

        with torch.no_grad():
            probs, hidden = decoder(inputs, prev_hidden)
        log_prob, indices = torch.topk(probs, beam_width)

        for node in range(batch_size):
            node_hidden = extract_hidden_from_batch(hidden, node)
            n = batch[node]
            for i in range(beam_width):
                # Quick fix to deal with different output dimensions of models
                if indices.size(0) == 1:
                    next_idx = indices[0, node, i].item()
                    log_p = log_prob[0, node, i].item()
                else:
                    next_idx = indices[node, 0, i].item()
                    log_p = log_prob[node, 0, i].item()

                next_node = BeamSearchNode(node_hidden, n, next_idx, n.log_prob + log_p, n.len + 1)
                next_nodes.put(next_node)

    node = BeamSearchNode(init_hidden, None, sos, 0, 1)
    nodes.put(node)

    decode_beams()

    while True:
        while next_nodes.qsize() > 0:
            nodes.put(next_nodes.get())
        next_nodes = PriorityQueue()
        if nodes.qsize() == 0:  # or nodes.qsize() > 2000:
            # Give up
            break
        if len(results) >= n_required:
            break
        decode_beams()

    if len(results) < top_k:
        results = [nodes.get(False) for _ in range(top_k)]

    utterances = []
    for n in sorted(results, key=lambda x: -x.eval()):
        utterance = [n.word_id]
        while n.prev is not None:
            n = n.prev
            utterance.append(n.word_id)
        utterance.reverse()
        utterances.append(utterance)

    return utterances


def extract_hidden_from_batch(hidden_batch: List[Tuple[torch.Tensor, torch.Tensor]], idx: int) -> List[
    Tuple[torch.Tensor, torch.Tensor]]:
    """
    Deal with the array of hidden states from the MoS model

    @param hidden_batch: Hidden states from a batch of inputs
    @param idx: Index of the input we care about
    @return: Hidden state for single input
    """
    node_hidden = [(layer_hidden[0][0, idx, :], layer_hidden[1][0, idx, :]) for layer_hidden in hidden_batch]
    return node_hidden
