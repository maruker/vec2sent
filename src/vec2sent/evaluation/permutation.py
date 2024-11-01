from tqdm import tqdm


def is_permutation(reference: str, hypothesis: str) -> bool:
    """
    Evaluate, whether two sentences are permutations of each other

    :param reference: Generated sentence
    :param hypothesis: Input sentence
    :return: True if the sentences contain exactly the same tokens.
    """
    reference = reference.split(' ')
    hypothesis = hypothesis.split(' ')

    return sorted(reference) == sorted(hypothesis)


def compute_fraction(filename, numerator_fn, denominator_fn):
    """
    Go through a file and compare each sentence pair

    :param filename: file to evaluate
    :param numerator_fn: boolean function to count in the numerator
    :param denominator_fn: boolean function to count in the denominator
    :return: fraction of n. of sentences fulfulling the numerator fn/ n. of sentences fulfilling the denominator fn
    """
    with open(filename, encoding='utf-8') as file:
        numerator = 0
        denominator = 0

        for i, line in enumerate(tqdm(file, total=3000)):
            line = line.lstrip().strip('\n')
            if i % 3 == 0:
                reference = line
            if i % 3 == 1:
                hypothesis = line
            if i % 3 == 2:
                numerator += int(numerator_fn(reference, hypothesis))
                denominator += int(denominator_fn(reference, hypothesis))
        return (numerator / denominator) * 100


def eval_permutations(filename):
    """
    :param filename: file to evaluate
    :return: fraction of permutations out of all sentences
    """
    return compute_fraction(
        filename,
        numerator_fn=is_permutation,
        denominator_fn=lambda x, y: True
    )


def eval_identity(filename):
    """
    :param filename: file to evaluate
    :return: fraction of identity out of all sentences
    """
    return compute_fraction(
        filename,
        numerator_fn=lambda x, y: x == y,
        denominator_fn=lambda x, y: True
    )


def eval_identity_permutation(filename):
    """
    :param filename: file to evaluate
    :return: fraction of identity out of all permutations
    """
    return compute_fraction(
        filename,
        numerator_fn=lambda x, y: x == y,
        denominator_fn=is_permutation
    )
