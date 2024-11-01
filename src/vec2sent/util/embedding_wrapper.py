from typing import Callable, List

import torch
from gensim.models import KeyedVectors
from bpemb import BPEmb

from nltk.tokenize import word_tokenize


class EmbeddingWrapper:

    def __init__(
            self,
            keyedvectors: KeyedVectors,
            lowercase: bool,
            device: torch.device,
            tokenizer: Callable[[str], List[str]] = None,
            oov_token: str = '<oov>'
    ):
        """
        @param keyedvectors: gensim word embedding object, containing vectors and dictionary
        @param lowercase: if true, the case of the words will be disregarded, all words will be looked up as lower case
        @param tokenizer: function, that splits input sentences into tokens
        @param oov_token: out of vocabulary symbol
        """
        self.keyedvectors = keyedvectors
        self.lowercase = lowercase
        self.encoder = tokenizer
        self.oov_token = oov_token

        vectors = torch.from_numpy(keyedvectors.vectors)
        self.torch_embeddings = torch.nn.Embedding.from_pretrained(vectors).to(device)
        self.decoder = torch.nn.Linear(keyedvectors.vectors.shape[1], keyedvectors.vectors.shape[0]).to(device)
        self.decoder.weight = self.torch_embeddings.weight
        self.torch_embeddings.weight.requires_grad = False
        self.decoder.weight.requires_grad = False

    def tokenize(self, sentence: str, start_token: str = None) -> List[str]:
        """
        Split a sentence into tokens that can be looked up in the vector space.
        Careful: If no tokenizer was given when the EmbeddingWrapper object was initialized, this function will assume
        that the text is already tokenized!

        @param sentence: sentence to tokenize
        @param start_token: token to add to the beginning of the sentence after tokenizing
        @return: list of tokens
        """
        if self.encoder is not None:
            tokens = self.encoder(sentence)
        else:
            if self.lowercase:
                sentence = sentence.lower()
            tokens = word_tokenize(sentence)

        if start_token is not None:
            return [start_token] + tokens
        else:
            return tokens

    def is_lowercase(self) -> bool:
        """
        @return: true, if the embeddings disregard word case
        """
        return self.lowercase

    def get_word(self, index: int) -> str:
        """
        @param index: word index in embedding space
        @return: string of the given word index
        """
        return self.keyedvectors.index2entity[index]

    def get_index(self, word: str) -> int:
        """
        @param word: string representation of a word
        @return: index of the given string in embedding space
        """
        return self.keyedvectors.vocab.get(word).index

    def get_end(self):
        """
        @return: index of end of sentence token in embedding space
        """
        return self.keyedvectors.vocab.get('</s>').index

    def get_oov(self):
        """
        @return: index of the out of vocabulary token
        """
        return self.keyedvectors.vocab.get(self.oov_token).index

    def get_embeddings(self) -> torch.nn.Embedding:
        """
        @return: pytorch embedding module
        """
        return self.torch_embeddings

    def get_size(self) -> int:
        """
        @return: size of each embedding vector
        """
        return self.keyedvectors.vectors.shape[1]

    def n_vocab(self):
        """
        @return: number of words in the embeddings
        """
        return self.keyedvectors.vectors.shape[0]

    def get_decoder(self) -> torch.nn.Module:
        """
        @return: matrix to compute inverse operation: embedding -> index
        """
        return self.decoder

    def __contains__(self, word: str) -> bool:
        """
        @param word: string representation of a word
        @return: true if the word is contained in the embeddings
        """
        return word in self.keyedvectors.vocab

    def embed(self, indices: torch.Tensor) -> torch.Tensor:
        """
        @param indices: Sequence of word indices
        @return: sequence of word embeddings
        """
        return self.torch_embeddings(indices)

    def normalize(self) -> None:
        """
        Normalize the length of every vector to 1
        """
        norms = torch.sum(self.torch_embeddings.weight ** 2, 1) ** (1 / 2)
        self.torch_embeddings.weight /= norms.view(norms.size(0), 1)
        self.decoder.weight = self.torch_embeddings.weight

    def get_weights(self) -> torch.Tensor:
        """
        @return: nxd word embedding matrix
        """
        return self.torch_embeddings.weight


def get_embeddings(path: str, lowercase: bool, device: torch.device) -> EmbeddingWrapper:
    """
    Load a word2vec txt file containing words and their vectors.

    @param path: filepath of the word embeddings
    @param lowercase: if true, case of input words will be disregarded
    @param device: where to move the embedding matrix
    @return: EmbeddingWrapper object
    """
    embeddings = KeyedVectors.load_word2vec_format(path)  # , limit=4000)
    return EmbeddingWrapper(embeddings, lowercase, device)


def get_bpemb(lang: str, dim: int, vocab_size: int, device: torch.device) -> EmbeddingWrapper:
    """
    Load byte pair encodings from the BPEmb library that fit the given lang, dim and vocab_size

    @param lang: Language of the embedding
    @param dim: Size of the individual embedding vectors
    @param vocab_size: Number of words in the embedding space
    @param device: Where to move the embedding matrix
    @return: EmbeddingWrapper object
    """
    bpe = BPEmb(lang=lang, dim=dim, vs=vocab_size)
    return EmbeddingWrapper(bpe.emb, False, device, tokenizer=bpe.encode, oov_token='<unk>')
