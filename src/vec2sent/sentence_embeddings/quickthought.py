from vec2sent.sentence_embeddings.abstract_sentence_embedding import AbstractSentenceEncoder
from vec2sent.quick_thought import encoder_manager
from vec2sent.quick_thought import configuration
from vec2sent.sentence_embeddings.cache_utils import get_quickthought_model_path, get_glove_embedding_path
import tensorflow as tf
import torch
import json
from typing import List


def get_model_configs() -> List[object]:
    tf.flags.DEFINE_string(
        "train_dir",
        get_quickthought_model_path())

    tf.flags.DEFINE_string(
        "Glove_path",
        get_glove_embedding_path())

    config_path = tf.flags.FLAGS.flag_values_dict()['model_config']

    with open(config_path) as json_config_file:
        model_configs = json.load(json_config_file)
    if not isinstance(model_configs, list):
        model_configs = [model_configs]
    model_configs = [configuration.model_config(c, mode="encode") for c in model_configs]
    return model_configs


class QuickThought(AbstractSentenceEncoder):
    def __init__(self, multichannel=True):
        model_configs = get_model_configs()

        if not multichannel:
            model_configs = [model_configs[-1]]
            self.size = 2400
        else:
            self.size = 4800

        self.manager = encoder_manager.EncoderManager()
        for c in model_configs:
            self.manager.load_model(c)
        pass

    def encode(self, sentences):
        embeddings = torch.from_numpy(self.manager.encode(sentences))
        return embeddings

    def to(self, device):
        return self

    def get_name(self):
        return 'quickthought'

    def get_size(self, embedding_size):
        return self.size

    def __del__(self):
        self.manager.close()
