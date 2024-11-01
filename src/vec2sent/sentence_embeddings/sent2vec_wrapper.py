from vec2sent.sentence_embeddings.abstract_sentence_embedding import AbstractSentenceEncoder
from vec2sent.sentence_embeddings.cache_utils import get_sent2vec_model_path
import sent2vec
import torch


class Sent2Vec(AbstractSentenceEncoder):
    def __init__(self):
        self.model = sent2vec.Sent2vecModel()
        self.model.load_model(get_sent2vec_model_path())

    def encode(self, sentences):
        embs = self.model.embed_sentences(sentences)
        return torch.from_numpy(embs)

    def get_name(self):
        return 'sent2vec'

    def get_size(self, embedding_size):
        return 700

    def to(self, device):
        return self
