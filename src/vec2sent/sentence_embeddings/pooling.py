import torch
from .abstract_sentence_embedding import AbstractPooling


class AvgMaxHier(AbstractPooling):
    def __init__(self):
        super().__init__()
        self.window = 3
        self.avgpoolwindow = torch.nn.AvgPool1d(self.window, stride=1)
        self.avgpool = torch.nn.AdaptiveAvgPool1d(1)
        self.maxpool = torch.nn.AdaptiveMaxPool1d(1)

    def forward(self, sentence):
        if sentence.size(2) < self.window:
            sentence1 = self.maxpool(sentence)
        else:
            sentence1 = self.maxpool(self.avgpoolwindow(sentence))
        sentence2 = self.avgpool(sentence)
        sentence3 = self.maxpool(sentence)

        sentence = torch.cat([sentence1, sentence2, sentence3], 1)
        return sentence.view(-1, sentence.size(1))

    def get_size(self, embedding_size):
        return embedding_size * 3

    def get_name(self):
        return "avgmaxhier"


class AvgHier(AbstractPooling):
    def __init__(self):
        super().__init__()
        self.window = 3
        self.avgpoolwindow = torch.nn.AvgPool1d(self.window, stride=1)
        self.avgpool = torch.nn.AdaptiveAvgPool1d(1)
        self.maxpool = torch.nn.AdaptiveMaxPool1d(1)

    def forward(self, sentence):
        if sentence.size(2) < self.window:
            sentence1 = self.maxpool(sentence)
        else:
            sentence1 = self.maxpool(self.avgpoolwindow(sentence))
        sentence2 = self.avgpool(sentence)

        sentence = torch.cat([sentence1, sentence2], 1)
        return sentence.view(-1, sentence.size(1))

    def get_size(self, embedding_size):
        return embedding_size * 2

    def get_name(self):
        return "avghier"


class Avg(AbstractPooling):
    def __init__(self):
        super().__init__()
        self.avgpool = torch.nn.AdaptiveAvgPool1d(1)

    def get_size(self, embedding_size):
        return embedding_size

    def forward(self, sentence):
        sentence = self.avgpool(sentence)
        return sentence.view(-1, sentence.size(1))

    def get_name(self):
        return "avg"


class Hier(AbstractPooling):
    def __init__(self):
        super().__init__()
        self.window = 3
        self.avgpool = torch.nn.AvgPool1d(self.window, stride=1)
        self.maxpool = torch.nn.AdaptiveMaxPool1d(1)

    def forward(self, sentence):
        if sentence.size(2) < self.window:
            sentence = self.maxpool(sentence)
        else:
            sentence = self.maxpool(self.avgpool(sentence))
        return sentence.view(-1, sentence.size(1))

    def get_size(self, embedding_size):
        return embedding_size

    def get_name(self):
        return "hier"
