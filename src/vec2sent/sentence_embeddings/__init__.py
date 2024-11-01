from typing import Dict, Any
from vec2sent.sentence_embeddings.pooling import Avg, Hier, AvgMaxHier
from vec2sent.sentence_embeddings.laser import LASER
from vec2sent.sentence_embeddings.infersent import InferSent
from vec2sent.sentence_embeddings.bert import FinetunedBERT, FinetunedBERTlarge
from vec2sent.sentence_embeddings.geometric_embedding import GEM
from vec2sent.sentence_embeddings.sent2vec_wrapper import Sent2Vec
from vec2sent.sentence_embeddings.quickthought import QuickThought


def get_sentence_embedding_by_name(name: str, options: Dict[str, Any] = None):
    if options is None:
        options = {}
    if name == "avg":
        return Avg(**options)
    if name == "laser":
        return LASER(**options)
    if name == "hier":
        return Hier(**options)
    if name == "avgmaxhier":
        return AvgMaxHier(**options)
    if name == "infersent":
        return InferSent(**options)
    if name == "sbert":
        return FinetunedBERT(**options)
    if name == "sbert-large":
        return FinetunedBERTlarge(**options)
    if name == "gem":
        return GEM(**options)
    if name == "sent2vec":
        return Sent2Vec(**options)
    if name == "quickthought":
        return QuickThought(**options)

    raise ValueError('Sentence embedding ' + name + ' does not exist')
