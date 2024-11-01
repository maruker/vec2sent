from abc import ABC

from vec2sent.sentence_embeddings.abstract_sentence_embedding import AbstractSentenceEncoder
from sentence_transformers import SentenceTransformer
import torch


class SBERTWrapper(AbstractSentenceEncoder, ABC):
    model: SentenceTransformer

    def encode(self, sentences):
        embeddings = self.model.encode(sentences, show_progress_bar=False)
        result = torch.FloatTensor(len(embeddings), embeddings[0].shape[0])
        for i in range(len(embeddings)):
            result[i] = torch.from_numpy(embeddings[i]).float()
        return result

    def to(self, device):
        self.model.to(device)
        return self


class FinetunedBERTlarge(SBERTWrapper):
    def __init__(self):
        self.model = SentenceTransformer('bert-large-nli-mean-tokens')
        pass

    def get_name(self):
        return 'sbert-large'

    def get_size(self, embedding_size):
        return 1024


class FinetunedBERT(SBERTWrapper):
    def __init__(self):
        self.model = SentenceTransformer('bert-base-nli-mean-tokens')

    def to(self, device):
        self.model.to(device)
        return self

    def get_name(self):
        return 'sbert'

    def get_size(self, embedding_size):
        return 768
