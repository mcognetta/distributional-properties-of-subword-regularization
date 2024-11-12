import random

# taken from https://github.com/tatHi/maxmatch_dropout

class MaxMatchTokenizer:
    def __init__(self, vocab=None, midPref='##', headPref=''):
        self.midPref = midPref
        self.headPref = headPref
        self.doNaivePreproc = False
        if vocab:
            self.__build(vocab)

    def __build(self, vocab):
        self.unkToken = '[UNK]'
        self.vocab = set(vocab)
        self.vocab.add(self.unkToken)
        self.vocabSize = len(self.vocab)
        
        self.maxLength = max(len(w) for w in self.vocab)

        self.word2id = {}
        self.id2word = {}
        for w in sorted(self.vocab):
            self.word2id[w] = len(self.word2id)
            self.id2word[self.word2id[w]] = w

    # This function corresponds to Algorithm 1 in the paper.
    def tokenizeWord(self, word, p=0.0):
        subwords = []
        i = 0
        wordLength = len(word)
        while i < wordLength:
            subword = None
            for j in range(1, min(self.maxLength+1, wordLength-i+1)):
                w = word[i:i+j]

                if 0==i: w = self.headPref + w
                if 0<i: w = self.midPref + w

                if w in self.vocab:
                    # random for subword regularization
                    if j==1 or p<random.random():
                        # drop acception with p
                        subword = w
                    
            if subword is None:
                # return unk if including unk
                return [self.unkToken]
            else:
                i += len(subword)-len(self.midPref) if 0<i else len(subword)-len(self.headPref)
                subwords.append(subword)
        return subwords

    def tokenize(self, text, p=0.0):
        if type(text)==list:
            return [self.tokenize(line, p) for line in text]
        if self.doNaivePreproc:
            text = self.naivePreproc(text)
        return [subword for word in text.split() for subword in self.tokenizeWord(word, p)]

    def encode(self, text, p=0.0):
        if type(text)==list:
            return [self.word2id[w] for line in text for w in self.tokenize(line, p)]
        return [self.word2id[w] for w in self.tokenize(text, p)]

    def loadVocab(self, path):
        words = [line.strip() for line in open(path)]
        self.vocab = set()
        self.word2id = {}
        self.id2word = {}
        for i, w in enumerate(words):
            self.vocab.add(w)
            self.word2id[w] = i
            self.id2word[i] = w
        self.vocabSize = len(self.vocab)
        self.maxLength = max(len(w) for w in self.vocab)

    def naivePreproc(self, text):
        return ' '.join(self.bertTokenizer.tokenize(text)).replace(' '+self.midPref, '')

if __name__=='__main__':
    vocab = '▁a ▁b ▁c ▁cc abc a b c'.split()
    sent = 'aabcb cda abcaacccc'
    print(vocab)
    print(sent)

    mmt = MaxMatchTokenizer(vocab, midPref='▁', headPref='')
    print(mmt.tokenize(sent))
    print(mmt.encode(sent))

    # print(mmt.tokenize([sent, sent]))
    # print(mmt.encode([sent, sent]))