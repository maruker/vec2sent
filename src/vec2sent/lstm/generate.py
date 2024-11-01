from typing import Dict, Tuple
import torch
import argparse
import logging

from vec2sent.lstm.model_utils import generate
from vec2sent.util.utils import de_bpemb_append
from vec2sent.sentence_embeddings import get_sentence_embedding_by_name
from vec2sent.lstm.contextual_mos_lstm import ConditionedRNNModel
from vec2sent.util.embedding_wrapper import get_bpemb
from vec2sent.dataset import load_dataset

from tqdm import tqdm


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Generate sentences by encoding each line in a file into sentence embeddings and then decoding with vec2sent')
    parser.add_argument("-c", "--checkpoint_path", type=str, required=True,
                        help="path or huggingface model repo of RNN decoder model checkpoint")
    parser.add_argument("-s", "--sentence_embedding", type=str, required=True,
                        help="Name of sentence embedding")
    parser.add_argument("-d", "--dataset_path", type=str, required=True,
                        help="path to dataset file containing one input string per line")
    parser.add_argument("-o", "--output_path", type=str, required=True,
                        help="path to where to write the outputs")
    parser.add_argument("-n", "--num_sentences", type=int, default=0,
                        help="limits the number of sentences loaded from the dataset if set to a number > 0")
    parser.add_argument("--device", type=str, default="cpu",
                        help="pytorch device. cpu, gpu or specified device index")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    device = torch.device(args.device)

    word_embeddings = get_bpemb('en', 300, 50000, device)
    model = ConditionedRNNModel.from_pretrained(args.checkpoint_path).to(device)

    out_file = open(args.output_path, 'w')

    sentence_embeddings = get_sentence_embedding_by_name(args.sentence_embedding,
                                                         options=model.config.get("options", None)).to(device)

    dataset = load_dataset(args.dataset_path, word_embeddings, sentence_embeddings, device,
                           True, 1, 120, args.num_sentences, start_token="<s>")

    del sentence_embeddings

    def generate_sentence(x: Dict[str, torch.Tensor]) -> Tuple[str, str, torch.Tensor]:
        """
        Returns the cross entropy loss for the generated sentence or None, if the iterator has no more elements

        @param x: datapoint from pytorch data generator
        @return input sentence, output sentence, loss
        """
        data, context, label = x['data'], x['context'], x['label'].to(device)

        # Compute loss for all tokens of the input sequence at once
        criterion = torch.nn.CrossEntropyLoss()
        output, _ = model(data, context)
        loss = criterion(output.contiguous().view(-1, output.size(2)), label.contiguous().view(-1)).item()

        # Print out generated and reference sentence
        sentence: List[str] = []
        for word_idx in data[0][1:]:
            if word_idx == word_embeddings.get_end():
                break
            de_bpemb_append(sentence, word_embeddings.get_word(word_idx))

        del output, _, data, label

        original_sentence = ' '.join(sentence)
        sentence = generate(model, word_embeddings, context, '<s>')
        generated_sentence = ' '.join(sentence)
        return original_sentence, generated_sentence, loss

    # -------------------------------------------------------------------------------
    # Generate sentences from all inputs, write them to the output file, compute loss
    # -------------------------------------------------------------------------------

    loss = 0
    i = 0

    sentences = list(tqdm(map(generate_sentence, dataset.loader), total=len(dataset.loader)))

    for original_sentence, generated_sentence, sentence_loss in sentences:
        out_file.write(original_sentence)
        out_file.write('\n')
        out_file.write(generated_sentence)
        out_file.write('\n')
        out_file.write('\n')

        loss += sentence_loss
        i += 1

    logger = logging.getLogger(__name__)
    logger.info('Average loss per sentence: {}'.format(loss / i))

    out_file.close()


if __name__ == "__main__":
    main()
