import logging

import torch
from typing import List


def n_parameters(model: torch.nn.Module) -> int:
    """
    Counts the total number of parameters in a model

    @param model: model to count
    @return: total number of parameters
    """
    logger = logging.getLogger(__name__)
    parameters = 0
    for name in model.state_dict():
        weight = model.state_dict()[name]
        logger.debug(name, '\t', weight.size())
        current_weight = 1
        for i in range(weight.dim()):
            current_weight *= weight.size(i)
        parameters += current_weight

    return parameters


def de_bpemb_append(sequence: List[str], token: str) -> List[str]:
    """
    Appends a new word to a generated text while concatenating byte pair encodings from BPEmb

    :param sequence: current generated sequence
    :param token: byte pair encoding to append

    :return: list of concatenated words
    """
    if token.startswith('▁'):
        sequence.append(token[1:])
    else:
        sequence[-1] += token

    return sequence
