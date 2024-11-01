from vec2sent.evaluation.mover_score import word_mover_score


def eval_mover(filename, device_name):
    """
    Go through file with references and generated sentences and calcualate the average mover score

    :param filename: file to evaluate
    :return: average mover score
    """
    references = []
    hypotheses = []

    with open(filename) as f:
        for i, line in enumerate(f):
            if i % 3 == 0:
                references.append(line.strip())
            if i % 3 == 1:
                hypotheses.append(line.strip())

    scores = word_mover_score(1, references, hypotheses, device=device_name)
    return sum(scores) * 100 / len(scores)  # Average score
