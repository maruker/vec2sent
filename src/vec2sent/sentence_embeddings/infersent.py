from vec2sent.sentence_embeddings.abstract_sentence_embedding import AbstractSentenceEncoder
from vec2sent.InferSent.models import InferSent as NakedInferSent
from vec2sent.sentence_embeddings.cache_utils import get_infersent_model_path, get_fasttext_embedding_path

import torch


class InferSent(AbstractSentenceEncoder):
    def __init__(self):
        model_path = get_infersent_model_path()
        params_model = {
            'bsize': 64,
            'word_emb_dim': 300,
            'enc_lstm_dim': 2048,
            'pool_type': 'max',
            'dpout_model': 0.0,
            'version': 2
        }
        infersent_model = NakedInferSent(params_model)
        infersent_model.load_state_dict(torch.load(model_path))
        infersent_model.set_w2v_path(get_fasttext_embedding_path())
        infersent_model.build_vocab_k_words(K=300000)

        self.model = infersent_model

    def encode(self, sentences):
        embeddings = torch.from_numpy(self.model.encode(sentences, tokenize=True))
        return embeddings

    def get_size(self, embedding_size):
        return 4096

    def get_name(self):
        return 'infersent'

    def to(self, device):
        self.model.to(device)
        return self

    def string_inputs(self):
        return True
