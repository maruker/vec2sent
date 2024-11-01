import subprocess
import os
from pathlib import Path


def eval_bleu(filename):
    """
    :param filename: file to evaluate with input and output lines
    :return: bleu score
    """
    src_file = open("bleu-reference", "w")
    tgt_file = open("bleu-hypothesis", "w")
    in_file = open(filename, "r")

    for i, line in enumerate(in_file):
        if i % 3 == 0:
            src_file.write(line.strip("\n"))
        if i % 3 == 1:
            tgt_file.write(line.strip("\n"))

    src_file.close()
    tgt_file.close()
    in_file.close()

    evaluation_folder = Path(__file__).resolve().parent
    bleu_script_path = str(evaluation_folder.joinpath("multi-bleu.perl"))
    process = subprocess.Popen("{} bleu-reference < bleu-hypothesis".format(bleu_script_path),
                               stdout=subprocess.PIPE, shell=True)

    scores = process.communicate()[0]

    os.remove('bleu-reference')
    os.remove('bleu-hypothesis')

    return scores
