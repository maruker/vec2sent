import argparse
import ast

from nltk.tokenize import word_tokenize
import torch

from vec2sent.sentence_embeddings import get_sentence_embedding_by_name
from vec2sent.lstm.contextual_mos_lstm import ConditionedRNNModel
from vec2sent.lstm.model_utils import generate
from vec2sent.util.embedding_wrapper import EmbeddingWrapper, get_bpemb


class ArithmeticExpressionVisitor(ast.NodeVisitor):
    """
    This class visits simple arithmetic expressions and throws an error if it encounters anything else.
    """

    def generic_visit(self, node: ast.AST) -> None:
        raise ValueError("Only simple arithmetic expressions are supported.")

    def visit_Add(self, node: ast.Add) -> None:
        ast.NodeVisitor.generic_visit(self, node)

    def visit_Sub(self, node: ast.Sub) -> None:
        ast.NodeVisitor.generic_visit(self, node)

    def visit_Mult(self, node: ast.Mult) -> None:
        ast.NodeVisitor.generic_visit(self, node)

    def visit_Div(self, node: ast.Div) -> None:
        ast.NodeVisitor.generic_visit(self, node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        ast.NodeVisitor.generic_visit(self, node)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
        ast.NodeVisitor.generic_visit(self, node)

    def visit_Num(self, node: ast.Num) -> None:
        ast.NodeVisitor.generic_visit(self, node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id not in ["a", "b", "c", "d", "e", "f"]:
            raise ValueError("Only variables a - f are supported.")


def compute_word_embeddings(sentence: str, word_embeddings: EmbeddingWrapper) -> torch.Tensor:
    sentence = " ".join(word_tokenize(sentence))
    words = word_embeddings.tokenize(sentence.strip("\n"))
    word_ids = torch.tensor([word_embeddings.get_index(w) for w in words], dtype=torch.long)
    return word_embeddings.embed(word_ids).unsqueeze(0).transpose(1, 2)


def main():
    parser = argparse.ArgumentParser(
        description="Perform arithmetic operations in the sentence vector space and decode the results")
    parser.add_argument("-s", "--sentence_embedding", type=str, required=True,
                        help="Name of sentence embedding")
    parser.add_argument("-c", "--checkpoint_path", type=str, required=True,
                        help="path or huggingface model repo of RNN decoder model checkpoint")
    parser.add_argument("-d", "--device", type=str, default="cpu",
                        help="pytorch device name")
    args = parser.parse_args()

    device = torch.device(args.device)

    sentence_embedding = get_sentence_embedding_by_name(args.sentence_embedding).to(device)
    word_embeddings = get_bpemb('en', 300, 50000, device)
    model = ConditionedRNNModel.from_pretrained(args.checkpoint_path).to(device)

    while True:
        for sentence_var in ["a", "b", "c", "d", "e", "f"]:
            sentence = input("Please enter sentence {} (Or nothing if done):".format(sentence_var))
            if sentence == "":
                if sentence_var == "a":
                    exit()
                break
            if sentence_embedding.input_strings():
                semb = sentence_embedding.encode([sentence])
                print(semb.shape)
            else:
                semb = sentence_embedding.encode(compute_word_embeddings(sentence, word_embeddings))
            exec("{} = semb".format(sentence_var))
        expr = input("Please enter an arithmetic expression (e.g. (a + b) * c / 2):")

        visitor = ArithmeticExpressionVisitor()
        visitor.visit(ast.parse(expr, mode="eval").body)
        result = eval(expr)

        generated_words = generate(model, word_embeddings, result, "<s>")
        print(" ".join(generated_words))


if __name__ == "__main__":
    main()
