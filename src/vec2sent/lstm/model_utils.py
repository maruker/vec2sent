from typing import List, Tuple

import torch

from vec2sent.util.embedding_wrapper import EmbeddingWrapper
from vec2sent.util.utils import de_bpemb_append
from vec2sent.lstm.beam_search import beam_decode


def get_model(
        context_size: int,
        hidden_size: int,
        num_layers: int,
        device: torch.device,
        path: str = None,
        torch_embeddings: torch.nn.Module = None
):
    """
    Load or create model with my hyperparameters

    @param context_size: Size of context vector (i.e. sentence embeddings)
    @param hidden_size: Size of hidden state vector
    @param num_layers: Number of hidden layers
    @param device: Where to load
    @param path: Model checkpoint path
    @param torch_embeddings: Word embedding vector matrix
    @return: Conditioned RNN model
    """
    from vec2sent.lstm.contextual_mos_lstm import ConditionedRNNModel, ModelConfig

    n_vocab = torch_embeddings.weight.size(0)
    emb_size = torch_embeddings.weight.size(1)

    model_config = ModelConfig(
        ntoken=n_vocab,
        ninp=emb_size,
        context_size=context_size,
        nhid=hidden_size,
        nhidlast=650,
        nlayers=num_layers,
        dropout=0.4,
        dropouth=0.2,
        dropouti=0.55,
        dropoute=0.1,
        wdrop=0.5,
        tie_weights=False,
        ldropout=0.29,
        n_experts=5
    )
    model = ConditionedRNNModel(model_config)
    model.init_embeddings(torch_embeddings)

    if path is not None:
        model.load_state_dict(torch.load(path, map_location=device))
    return model


def generate(
        model,
        embeddings: EmbeddingWrapper,
        context: torch.Tensor,
        start_token: str,
        hidden=None
) -> List[str]:
    """
    Generate sequence using beam search.

    @param model: Conditioned RNN model
    @param embeddings: Word embeddings
    @param context: Context vector (i.e. sentence embedding)
    @param start_token: start of sequence token
    @param hidden: initial hidden state
    @return: Generated sequence as list of word ids
    """
    start = embeddings.get_index(start_token)
    end = embeddings.get_end()

    def stack_hidden_states(
            hidden_states: List[Tuple[torch.Tensor, torch.Tensor]],
            batch_size: int
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Hidden states come in as a list of tensors and need to be combined into tensors

        @param hidden_states: List of hidden states
        @param batch_size: Batch size
        @return: Tuple containing (hidden, cell) all stacked together
        """
        concatenated_hidden_states = []
        for i in range(model.nlayers):
            hidden_init, cell_init = zip(*[h[i] for h in hidden_states])
            concatenated_hidden_states.append((
                torch.stack(hidden_init, ).view(1, batch_size, -1),
                torch.stack(cell_init, ).view(1, batch_size, -1)
            ))
        return concatenated_hidden_states

    def decoder(
            inputs: List[int],
            hidden_states: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Run RNN model on current inputs and hidden states

        @param inputs: List of word indices
        @param hidden_states: List of hidden states as tuples of (hidden, cell)
        @return: tuple of (outputs, hidden_states)
        """
        batch_size = len(inputs)
        inputs = [torch.tensor([x], dtype=torch.long) for x in inputs]
        if hidden_states == [None]:
            hidden_states = None
        else:
            hidden_states = stack_hidden_states(hidden_states, batch_size)
        return model(inputs, [context[0].view(-1)] * len(inputs), hidden=hidden_states, return_prob=True)

    output = beam_decode(decoder, hidden, start, end, 100)
    words: List[str] = []
    best_output = output[0]  # Highest probability from beam search
    best_output = best_output[1:-1]  # Remove sos and eos
    for idx in best_output:
        de_bpemb_append(words, embeddings.get_word(idx))

    return words
