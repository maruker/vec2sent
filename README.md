# Vec2Sent<br><sub><sup>Probing Sentence Embeddings with Natural Language Generation</sup></sub>
[![arXiv](https://img.shields.io/badge/View%20on%20arXiv-B31B1B?logo=arxiv&labelColor=gray)](https://arxiv.org/abs/2011.00592)

We introspect black-box sentence embeddings by conditionally generating from them with the
objective to retrieve the underlying discrete sentence. We perceive of this as a new unsupervised
probing task and show that it correlates well with downstream task performance. We also illustrate
how the language generated from different encoders differs. We apply our approach to generate
sentence analogies from sentence embeddings.

## Quickstart

You can quickly install Vec2Sent using pip:

```shell
pip install "vec2sent @ git+https://github.com/maruker/vec2sent.git"
```

There are three entry points to **generate** and **evaluate** sentences, and to perform **arithmetic** in the vector space.

### Vector Arithmetic

```shell
vec2sent_arithmetic -s infersent -c maruker/vec2sent-infersent
```

```text
Please enter sentence a (Or nothing if done):his name is robert
Please enter sentence b (Or nothing if done):he is a doctor
Please enter sentence c (Or nothing if done):her name is julia
Please enter sentence d (Or nothing if done):
Please enter an arithmetic expression (e.g. (a + b) * c / 2):b-a+c
 she is a doctor
```

### Sentence Generation

For example, generate outputs using the hierarchical sentence embedding

```shell
vec2sent_generate -s hier -c maruker/vec2sent-hier -d data/test.en.2008 -o hier.txt
```

### Evaluation

The outputs from the previous step can now be evaluated. For example, the following command computes the bleu score

```shell
vec2sent_evaluate --metric BLEU --file hier.txt
```

The following metrics are available

| Parameter | Explanation                                                                       |
|-----------|-----------------------------------------------------------------------------------|
| ID        | Fraction of all sentences where the output is identical to the input              |
| PERM      | Fraction of all output sentences that can be formed as a permutation of the input |
| ID_PERM   | Fraction of all permuations that are identical to the input                       |
| BLEU      | Document BLEU score                                                               |
| MOVER     | Average Mover Score between input and output sentences                            |


> [!TIP]
> Vec2Sent needs to download several gigabites of sentence embedding models. Those files can be deleted using the command `vec2sent_cleanup`

## Available Models

We upload our models to the Hugging Face Hub. The following table shows, which parameters to set in order to load the sentence embeddings and corresponding Vec2Sent models.

| Sentence embedding name `-s` | Checkpoint `-c`              | Explanation                                                                       |
|------------------------------|------------------------------|-----------------------------------------------------------------------------------|
| avg                          | maruker/vec2sent-avg         | Average pooling on [BPEmb](https://github.com/bheinzerling/bpemb) word embeddings |
| hier                         | maruker/vec2sent-hier        | Hierarchical pooling on [BPEmb](https://github.com/bheinzerling/bpemb)            |
| gem                          | maruker/vec2sent-gem         | [Geometric Embeddings](https://github.com/fursovia/geometric_embedding)           |
| sent2vec                     | maruker/vec2sent-sent2vec    | [Sent2Vec](https://github.com/epfml/sent2vec)                                     |
| infersent                    | maruker/vec2sent-infersent   | [InferSent](https://github.com/facebookresearch/InferSent)                        |
| sbert-large                  | maruker/vec2sent-sbert-large | [SBERT](https://github.com/UKPLab/sentence-transformers)                          |

Additional sentence embeddings can be used by extending the class ``vec2sent.sentence_embeddings.abstract_sentence_embedding.AbstractEmbedding``.

## Installation

#### (Optional) Setup Virtual Environment

```shell
python -m venv venv
source venv/bin/activate
```

#### Download requirements
```shell
# Download git submodules (MoS model and some sentence embeddings)
git submodule update --init
```

#### Install

```shell
pip install .
```

## Citation
If you find Vec2Sent useful in your academic work, please consider citing
```
@inproceedings{kerscher-eger-2020-vec2sent,
    title = "{V}ec2{S}ent: Probing Sentence Embeddings with Natural Language Generation",
    author = "Kerscher, Martin  and
      Eger, Steffen",
    booktitle = "Proceedings of the 28th International Conference on Computational Linguistics",
    month = dec,
    year = "2020",
    address = "Barcelona, Spain (Online)",
    publisher = "International Committee on Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.coling-main.152",
    pages = "1729--1736",
    abstract = "We introspect black-box sentence embeddings by conditionally generating from them with the objective to retrieve the underlying discrete sentence. We perceive of this as a new unsupervised probing task and show that it correlates well with downstream task performance. We also illustrate how the language generated from different encoders differs. We apply our approach to generate sentence analogies from sentence embeddings.",
}
```

## Acknowledgments

The models are based on [Mixture of Softmaxes](https://github.com/zihangdai/mos) with a context vector added to the inputs.
