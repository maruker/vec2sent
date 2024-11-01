from abc import ABC, abstractmethod
from typing import Union
import torch.nn as nn
import torch


class AbstractEmbedding(ABC):

    @abstractmethod
    def encode(self, sentence: Union[str, torch.Tensor]) -> torch.Tensor:
        """
        This function should be called to generate sentence embeddings,
        whether the implementation is a torch module or a wrapper for a sentence embedding.

        @param sentence: Input sentence.
        @return: Sentence embedding.
        """
        pass

    @abstractmethod
    def get_size(self, word_embedding_size: int) -> int:
        """
        Calculate the size of the sentence embedding vectors.

        @param word_embedding_size: size of inputs (only used for pooling)

        @return sentence embedding size
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """
        @return name of sentence embeddings
        """
        pass

    @abstractmethod
    def input_strings(self) -> bool:
        """
        @return True if the input to the encode method should be strings
        """
        pass

    def get(self, sentence) -> torch.FloatTensor:
        embedding = self.encode(sentence)
        if torch.isnan(embedding).any():
            raise ValueError('NaN in sentence embedding')
        if isinstance(self, AbstractSentenceEncoder):
            assert embedding.size(-1) == self.get_size(0)
        return embedding


class AbstractPooling(AbstractEmbedding, nn.Module, ABC):
    """
    Implement the embedding in the forward function
    (basically, write a normal Pytorch Module)
    """

    def encode(self, sentence: torch.FloatTensor) -> torch.FloatTensor:
        embedding = self(sentence)
        assert embedding.size(-1) == self.get_size(sentence.size(-2))
        return embedding

    def input_strings(self):
        return False


class AbstractSentenceEncoder(AbstractEmbedding, ABC):

    @abstractmethod
    def encode(self, sentence: str) -> torch.FloatTensor:
        """
        Implement embedding in this method
        """
        pass

    @abstractmethod
    def to(self, device: torch.device) -> AbstractEmbedding:
        pass

    def input_strings(self):
        return True
