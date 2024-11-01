from tqdm import tqdm


def eval_length(filename):
    """
    Evaluate length difference between input and output sentences

    :param filename: file to evaluate
    :return: mean length distance
    """
    difference = 0
    with open(filename) as file:
        for i, line in enumerate(tqdm(file, total=30000)):
            if i % 3 == 0:
                ref = line.strip("\n").split(" ")
            if i % 3 == 1:
                hyp = line.strip("\n").split(" ")
            if i % 3 == 2:
                difference += abs(len(ref) - len(hyp))
    return difference / 10000
