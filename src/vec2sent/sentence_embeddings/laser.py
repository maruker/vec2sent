from vec2sent.sentence_embeddings.abstract_sentence_embedding import AbstractSentenceEncoder, AbstractEmbedding
from laserembeddings import Laser
import torch


class LASER(AbstractSentenceEncoder):
    def __init__(self):
        super().__init__()
        self.encoder = Laser()

    def encode(self, sentence: str) -> torch.FloatTensor:
        embeddings = self.encoder.embed_sentences([sentence], lang="en")
        return torch.from_numpy(embeddings, type=torch.float)

    def to(self, device: torch.device) -> AbstractEmbedding:
        return self

    def get_size(self, word_embedding_size: int) -> int:
        return 1024

    def get_name(self) -> str:
        return "laser"
