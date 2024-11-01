import logging
from tqdm import tqdm
from nltk.tokenize import word_tokenize

from vec2sent.sentence_embeddings.abstract_sentence_embedding import AbstractEmbedding
from vec2sent.util.embedding_wrapper import EmbeddingWrapper
from vec2sent.dataset.sentence_dataset import SortedSentenceDataset

import torch


def determine_batch_size(sentence_embeddings: AbstractEmbedding) -> int:
    if sentence_embeddings.get_name() in ['randomLSTM', 'borep', 'gem'] or sentence_embeddings.input_strings():
        return 200
    return 1


def load_dataset(
        path: str,
        word_embeddings: EmbeddingWrapper,
        sentence_embeddings: AbstractEmbedding,
        device: torch.device,
        leave_order: bool,
        batch_size: int,
        max_len: int,
        num_sentences: int = 0,
        start_token: str = None
) -> SortedSentenceDataset:
    """
    Loads a dataset for training or evaluation.

    @param path: path to the file containing the data
    @param word_embeddings: word embeddings
    @param sentence_embeddings: sentence embeddings
    @param device: where to initially load the dataset (might not fit in VRAM)
    @param leave_order: whether to leave the dataset in order, or sort it into batches of even length
    @param batch_size: batch size
    @param max_len: Maximum sentence length in dataset
    @param num_sentences: if set to a number > 0, the dataset will be cut off after said number of sentences
    @param start_token: String added to each line of the dataset (after tokenization)
    """

    dataset = SortedSentenceDataset(word_embeddings, sentence_embeddings, device, batch_size, max_len)
    oov = 0

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(tqdm(f, total=num_sentences, desc="Loading dataset {}".format(path))):
            # Apply basic tokenization to punctuation first
            line = " ".join(word_tokenize(line))
            line = line.replace("''", '"').replace("``", '"')

            oov += dataset.add(line, max_len, start_token)

            if i == num_sentences - 1:
                break

    logger = logging.getLogger(__name__)
    logger.info('Loaded {} sentences'.format(len(dataset)))
    dataset.finish_init()

    logger.info("Out of vocabulary: {}".format(oov))

    pin_memory = device.type != 'cpu'
    dataset.create_data_loader(batch_size, leave_order, pin_memory)
    return dataset
