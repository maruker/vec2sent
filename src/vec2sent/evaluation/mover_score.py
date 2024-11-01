from __future__ import division
import torch
import numpy as np
from tqdm import tqdm
from pyemd import emd

from pytorch_pretrained_bert import BertTokenizer, BertModel
from collections import defaultdict


def padding(arr, pad_token, dtype=torch.long):
    lens = torch.LongTensor([len(a) for a in arr])
    max_len = lens.max().item()
    padded = torch.ones(len(arr), max_len, dtype=dtype) * pad_token
    mask = torch.zeros(len(arr), max_len, dtype=torch.long)
    for i, a in enumerate(arr):
        padded[i, :lens[i]] = torch.tensor(a, dtype=dtype)
        mask[i, :lens[i]] = 1
    return padded, lens, mask


def bert_encode(model, x, attention_mask):
    model.eval()
    x_seg = torch.zeros_like(x, dtype=torch.long)
    with torch.no_grad():
        x_encoded_layers, pooled_output = model(x, x_seg, attention_mask=attention_mask, output_all_encoded_layers=True)
    return x_encoded_layers


def collate_idf(arr, tokenize, numericalize, idf_dict,
                pad="[PAD]", device='cuda'):
    tokens = [["[CLS]"] + tokenize(a) + ["[SEP]"] for a in arr]
    arr = [numericalize(a) for a in tokens]

    idf_weights = [[idf_dict[i] for i in a] for a in arr]

    pad_token = numericalize([pad])[0]

    padded, lens, mask = padding(arr, pad_token, dtype=torch.long)
    padded_idf, _, _ = padding(idf_weights, pad_token, dtype=torch.float)

    return padded, padded_idf, lens, mask, tokens


def get_bert_embedding(all_sens, model, tokenizer, idf_dict,
                       batch_size=-1, device='cuda'):
    padded_sens, padded_idf, lens, mask, tokens = collate_idf(all_sens,
                                                              tokenizer.tokenize, tokenizer.convert_tokens_to_ids,
                                                              idf_dict,
                                                              device=device)

    if batch_size == -1: batch_size = len(all_sens)

    embeddings = []
    with torch.no_grad():
        for i in range(0, len(all_sens), batch_size):
            batch_embedding = bert_encode(model, padded_sens[i:i + batch_size],
                                          attention_mask=mask[i:i + batch_size])
            batch_embedding = torch.stack(batch_embedding)
            embeddings.append(batch_embedding)
            del batch_embedding

    total_embedding = torch.cat(embeddings, dim=-3)
    return total_embedding, lens, mask, padded_idf, tokens


def word_mover_score(ngram, refs, hyps, batch_size=256, device='cuda:0'):
    model_name = 'bert-base-uncased'
    device = 'cpu'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    model.eval()

    idf_dict_ref = defaultdict(lambda: 1.)
    idf_dict_hyp = defaultdict(lambda: 1.)
    preds = []
    for batch_start in tqdm(range(0, len(refs), batch_size)):
        batch_refs = refs[batch_start:batch_start + batch_size]
        batch_hyps = hyps[batch_start:batch_start + batch_size]

        ref_embedding, ref_lens, ref_masks, ref_idf, ref_tokens = get_bert_embedding(batch_refs, model, tokenizer,
                                                                                     idf_dict_ref,
                                                                                     device=device)
        hyp_embedding, hyp_lens, hyp_masks, hyp_idf, hyp_tokens = get_bert_embedding(batch_hyps, model, tokenizer,
                                                                                     idf_dict_hyp,
                                                                                     device=device)

        ref_embedding.div_(torch.norm(ref_embedding, dim=-1).unsqueeze(-1))
        hyp_embedding.div_(torch.norm(hyp_embedding, dim=-1).unsqueeze(-1))

        ref_embedding_max, _ = torch.max(ref_embedding[-5:], dim=0, out=None)
        hyp_embedding_max, _ = torch.max(hyp_embedding[-5:], dim=0, out=None)

        ref_embedding_min, _ = torch.min(ref_embedding[-5:], dim=0, out=None)
        hyp_embedding_min, _ = torch.min(hyp_embedding[-5:], dim=0, out=None)

        ref_embedding_avg = ref_embedding[-5:].mean(0)
        hyp_embedding_avg = hyp_embedding[-5:].mean(0)

        ref_embedding = torch.cat([ref_embedding_min, ref_embedding_avg, ref_embedding_max], -1)
        hyp_embedding = torch.cat([hyp_embedding_min, hyp_embedding_avg, hyp_embedding_max], -1)

        num_refs = len(ref_embedding)

        for i in range(num_refs):
            ref_ids = range(0, len(ref_tokens[i]))
            hyp_ids = range(0, len(hyp_tokens[i]))

            ref_embedding_i, ref_idf_i = load_ngram(ref_ids, ref_embedding[i], ref_idf[i], ngram, 1, device)
            hyp_embedding_i, hyp_idf_i = load_ngram(hyp_ids, hyp_embedding[i], hyp_idf[i], ngram, 1, device)

            raw = torch.cat([ref_embedding_i, hyp_embedding_i], 0)
            raw.div_(torch.norm(raw, dim=-1).unsqueeze(-1) + 0.000001)

            distance_matrix = pairwise_distances(raw, raw)

            c1 = np.zeros(len(ref_idf_i) + len(hyp_idf_i), dtype=np.double)
            c2 = np.zeros(len(ref_idf_i) + len(hyp_idf_i), dtype=np.double)

            c1[:len(ref_idf_i)] = ref_idf_i
            c2[-len(hyp_idf_i):] = hyp_idf_i

            c1 = _safe_divide(c1, np.sum(c1))
            c2 = _safe_divide(c2, np.sum(c2))
            score = 1 - emd(c1, c2, distance_matrix.double().cpu().numpy())
            preds.append(score)
    return preds


def slide_window(a, w=3, o=2):
    if a.size - w + 1 <= 0:
        w = a.size
    sh = (a.size - w + 1, w)
    st = a.strides * 2
    view = np.lib.stride_tricks.as_strided(a, strides=st, shape=sh)[0::o]
    return view.copy().tolist()


def _safe_divide(numerator, denominator):
    return numerator / (denominator + 0.00001)


def load_ngram(ids, embedding, idf, n, o, device):
    new_a = []
    new_idf = []

    slide_wins = slide_window(np.array(ids), w=n, o=o)
    for slide_win in slide_wins:
        new_idf.append(idf[slide_win].sum().item())
        scale = _safe_divide(idf[slide_win], idf[slide_win].sum(0)).unsqueeze(-1)
        tmp = (scale * embedding[slide_win]).sum(0)
        new_a.append(tmp)
    new_a = torch.stack(new_a, 0).to(device)
    return new_a, new_idf


def pairwise_distances(x, y=None):
    x_norm = (x ** 2).sum(1).view(-1, 1)
    y_norm = (y ** 2).sum(1).view(1, -1)
    y_t = torch.transpose(y, 0, 1)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    return torch.clamp(dist, 0.0, np.inf)
