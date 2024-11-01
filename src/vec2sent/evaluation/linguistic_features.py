import spacy
from tqdm import tqdm


class LingFeatures:

    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')

    def abs_intersection(self, l1, l2):
        count = 0
        l2 = l2.copy()
        for x in l1:
            if x in l2:
                count += 1
                l2.remove(x)
        return count

    def compare_subject_number(self, doc1, doc2):

        is_plural = lambda x: (x.tag_[-1] == 'S' or x.tag_ == 'PRP' and x.text in ['we', 'We', 'They', 'they'])
        extract_subjn = lambda doc: list(map(lambda x: int(is_plural(x)), filter(lambda x: 'subj' in x.dep_, doc)))

        subjn1 = extract_subjn(doc1)
        subjn2 = extract_subjn(doc2)

        try:
            return self.abs_intersection(subjn1, subjn2) / len(subjn1)
        except ZeroDivisionError:
            return 1

    def compare_tense(self, doc1, doc2):
        extract_verbs = lambda doc: list(filter(lambda x: x.startswith('V'), map(lambda x: x.tag_, doc)))

        verbs1 = extract_verbs(doc1)
        verbs2 = extract_verbs(doc2)

        try:
            return self.abs_intersection(verbs1, verbs2) / len(verbs1)
        except ZeroDivisionError:
            return 1

    def eval(self, filename):
        subn = 0
        tense = 0
        with open(filename) as file:
            for i, line in enumerate(tqdm(file, total=30000)):
                if i % 3 == 0:
                    ref = line.strip("\n")
                if i % 3 == 1:
                    hyp = line.strip("\n")
                if i % 3 == 2:
                    doc1 = self.nlp(ref)
                    doc2 = self.nlp(hyp)
                    subn += self.compare_subject_number(doc1, doc2)
                    tense += self.compare_tense(doc1, doc2)

        return {'Subject number': subn, 'Tense': tense}
