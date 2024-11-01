import argparse
from vec2sent.evaluation.permutation import eval_permutations, eval_identity, eval_identity_permutation
from vec2sent.evaluation.bleu.eval_bleu import eval_bleu
from vec2sent.evaluation.eval_mover import eval_mover


def main():
    parser = argparse.ArgumentParser(description='Compute different metrics on a file with generated sentences.\
                            The file must have the following layout: reference line, generated line, empty line, repeat')

    parser.add_argument('--metric', dest='metric',
                        help='Evaluation metric. Choose from BLEU, MOVER, ID, PERM, ID_PERM, perplexity', required=True)
    parser.add_argument('--file', dest='filepath', help='File to run evaluation on', required=True)
    parser.add_argument("--device", help="Device to run evaluation on (only used by mover score)", default="cpu")

    args = parser.parse_args()

    if args.metric == 'PERM':
        print('Percentage of nontrivial permutations: {}'
              .format(eval_permutations(args.filepath)))
    if args.metric == 'ID':
        print('Percentage of sentences identical to input: {}'
              .format(eval_identity(args.filepath)))
    if args.metric == 'ID_PERM':
        print('Percentage of permutations that are identical to the input: {}'
              .format(eval_identity_permutation(args.filepath)))
    if args.metric == 'BLEU':
        print('BLEU: {}'
              .format(eval_bleu(args.filepath)))
    if args.metric == 'MOVER':
        print('MOVER: {}'
              .format(eval_mover(args.filepath, args.device)))


if __name__ == "__main__":
    main()

