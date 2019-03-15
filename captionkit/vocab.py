'''Author: Brandon Trabucco, Copyright 2018
Vocabulary object for converting words to ids.'''


class Vocabulary(object):
    def __init__(self, vocab_names, start_word, end_word, unk_word):
        vocab = dict([(x, y) for (y, x) in enumerate(vocab_names)])
        print("Created vocabulary with %d names." % len(vocab_names))
        self.vocab = vocab
        self.reverse_vocab = vocab_names
        self.start_id = vocab[start_word]
        self.end_id = vocab[end_word]
        self.unk_id = vocab[unk_word]
    def word_to_id(self, word):
        if isinstance(word, list):
            return [self.word_to_id(w) for w in word]
        if word not in self.vocab:
            return self.unk_id
        return self.vocab[word]
    def id_to_word(self, index):
        if isinstance(index, list):
            return [self.id_to_word(i) for i in index]
        if index < 0 or index >= len(self.reverse_vocab):
            return self.reverse_vocab[self.unk_id]
        return self.reverse_vocab[index]