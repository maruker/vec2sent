from vec2sent.sentence_embeddings.abstract_sentence_embedding import AbstractSentenceEncoder
from vec2sent.geometric_embedding import SentenceEmbedder
from bpemb import BPEmb
import torch


class GEM(AbstractSentenceEncoder):
    def __init__(self):
        bpe_embeddings = BPEmb(lang='en', vs=50000, dim=300)
        self.vocab = {word: i for i, word in enumerate(bpe_embeddings.emb.index2entity)}
        self.vocab['UNKNOWN_TOKEN'] = self.vocab['<unk>']
        self.vectors = bpe_embeddings.vectors

    def encode(self, sentences):
        gem = SentenceEmbedder(sentences, self.vectors, self.vocab)
        return torch.from_numpy(gem.gem()).float()

    def get_size(self, embedding_size):
        return 300

    def get_name(self):
        return "gem"

    def to(self, device):
        return self
