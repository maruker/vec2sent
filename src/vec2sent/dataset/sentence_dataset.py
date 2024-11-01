from random import shuffle
from typing import Dict, List

import torch
from torch.utils.data import Dataset, Sampler, DataLoader

from vec2sent.util.embedding_wrapper import EmbeddingWrapper
from vec2sent.sentence_embeddings.abstract_sentence_embedding import AbstractEmbedding


class SortedSentenceDataset(Dataset):
    """
    Sentence dataset that is sorted into batches of similar length and then shuffled.
    Loads word embedding indices and sentence embeddings.
    """

    def __init__(
            self,
            word_embeddings: EmbeddingWrapper,
            sentence_embeddings: AbstractEmbedding,
            device: torch.device,
            batch_size: int,
            max_len: int
    ) -> None:
        """
        Sentence dataset that is sorted into batches of similar length and then shuffled.
        Loads word embedding indices and sentence embeddings.

        @param word_embeddings: Word embeddings
        @param sentence_embeddings: Sentence embeddings
        @param device: Where to initially load dataset (might not fit into VRAM)
        @param batch_size: Batch size
        @param max_len: Max sentence length
        """
        self.loader = None
        self.word_embeddings = word_embeddings
        self.sentence_embeddings = sentence_embeddings
        self.device = device
        self.batch_size = batch_size
        self.i = 0
        self.embedding_matrix = word_embeddings.get_embeddings().weight
        self.inputs: List[torch.LongTensor] = []
        self.lm_contexts: List[torch.FloatTensor] = []
        self.lengths: List[int] = []
        self.sentence_batch: List[str] = []
        self.sentence: torch.Tensor = torch.zeros((batch_size, self.word_embeddings.get_size(), max_len - 2)).to(device)

    def add(self, sentence: str, max_len: int, start_token: str = None) -> int:
        """
        Adds a sentence to the dataset. Throws an exception if the sentence is longer than max_len.

        @param sentence: Sentence to add
        @param max_len: Max sentence length
        @param start_token: Start token to add to the beginning of each sentence
        @return: Number of out of vocabulary words in the sentence
        """
        words = sentence.strip('\n')
        words = self.word_embeddings.tokenize(words, start_token)
        sentence_length = min(len(words) + 1, max_len)
        self.inputs.append(torch.zeros((sentence_length,), dtype=torch.long))
        oov = 0

        # Set to exact sentence length -> don't include any padding
        if self.batch_size == 1:
            self.sentence = torch.zeros((self.batch_size, self.word_embeddings.get_size(), sentence_length - 2)).to(
                self.device)

        assert len(words) <= max_len
        j = 0
        for word in words:
            if j == max_len - 1:
                break
            if word in self.word_embeddings:
                # Add to input and sentence embedding, dont encode start tokens
                self.inputs[-1][j] = self.word_embeddings.get_index(word)
                if not self.sentence_embeddings.input_strings() and j > 0:
                    self.sentence[self.i % self.batch_size, :, j - 1] = self.embedding_matrix[self.inputs[-1][j]]
            else:
                oov += 1
                self.inputs[-1][j] = self.word_embeddings.get_oov()
                self.sentence[self.i % self.batch_size, :, j - 1] = 0
            j += 1
        self.inputs[-1][j] = self.word_embeddings.get_end()
        self.lengths.append(sentence_length)

        if self.sentence_embeddings.input_strings():
            self.sentence_batch.append(sentence.strip('\n'))
        if self.i % self.batch_size == self.batch_size - 1:
            with torch.no_grad():
                self.calculate_sentence_embeddings(self.batch_size)

        self.i += 1
        return oov

    def calculate_sentence_embeddings(self, num_sentences: int) -> None:
        """
        Calculate sentence embeddings for current batch.

        @param num_sentences: Only process sentence 1 - num_sentences, in case the dataset ends in the middle of a batch
        """
        if self.sentence_embeddings.input_strings():
            self.lm_contexts.append(self.sentence_embeddings.get(self.sentence_batch).cpu())
            self.sentence_batch = []
        else:
            self.lm_contexts.append(self.sentence_embeddings.get(self.sentence[:num_sentences]).cpu())

    def create_data_loader(self, batch_size: int, leave_order: bool, pin_memory: bool):
        """
        Create a torch DataLoader around this torch Dataset.

        @param batch_size: Batch size
        @param leave_order: If false, sentences are shuffled and sorted into batches of approximately equal length
        @param pin_memory: See documentation of torch.utils.data.DataLoader
        """
        sampler = self.get_sampler(batch_size, leave_order)
        self.loader = DataLoader(
            self,
            batch_size=1,
            sampler=sampler,
            collate_fn=lambda x: x[0],
            pin_memory=pin_memory,
            num_workers=0
        )

    def finish_init(self):
        """
        In case the dataset ended in the middle of processing a batch, finish processing the last batch
        """
        if not self.i % self.batch_size == 0:
            self.calculate_sentence_embeddings(self.i % self.batch_size)
        self.lm_contexts = torch.cat(self.lm_contexts, 0)
        del self.sentence_embeddings

    def get_sampler(self, batch_size: int, leave_order: bool) -> "BatchSampler":
        """
        Create sampler that batches sentences of similar length together and shuffles the batches afterward.

        @param batch_size: Batch size
        @param leave_order: If True, the sampler neither groups by length nor shuffles the sentences.
        @return: Batch sampler
        """
        return BatchSampler(dict(enumerate(self.lengths)), batch_size, leave_order)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, indices):
        # Pack up batches and pad with end indices
        input_data = [self.inputs[i][:-1] for i in indices]
        label = [self.inputs[i][1:] for i in indices]
        context = [self.lm_contexts[i] for i in indices]

        label = torch.nn.utils.rnn.pad_sequence(label)

        return {
            'data': input_data,
            'context': context,
            'label': label
        }


class BatchSampler(Sampler):
    """
    BatchSampler creates batches of indices of sentences with similar lengths and shuffles the batches afterward.
    """

    def __init__(self, lengths: Dict[int, int], batch_size: int, leave_order: bool):
        """
        BatchSampler creates batches of indices of sentences with similar lengths and shuffles the batches afterward.

        @param lengths: Dictionary mapping from dataset index to sentence length.
        @param batch_size: Batch size
        @param leave_order: If set to True, the sentences are neither grouped by length nor shuffled.
        """
        self.lengths = list(lengths.items())
        if not leave_order:
            shuffle(self.lengths)
            self.lengths = sorted(self.lengths, key=lambda kv: kv[1], reverse=True)
        self.leave_order = leave_order
        self.batch_size = batch_size

    def __iter__(self):
        num_batches = len(self.lengths) // self.batch_size
        if not len(self.lengths) % self.batch_size == 0:
            num_batches += 1

        # Group indices of -batch_size- sentences with similar lengths
        indices = [[kv[0] for kv in self.lengths[i * self.batch_size:i * self.batch_size + self.batch_size]] for i in
                   range(num_batches)]
        if not self.leave_order:
            shuffle(indices)
        return iter(indices)

    def __len__(self):
        return len(self.lengths)
