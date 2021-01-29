class TokenDict:
    def __init__(self):
        self.vocab = dict()
        self.cnt = 0

    def check(self, token):
        if token not in self.vocab:
            return None
        else:
            return str(self.vocab[token])

    def get(self, token):
        if token not in self.vocab:
            self.vocab[token] = self.cnt
            self.cnt += 1
        return str(self.vocab[token])

    def save(self, output_path):
        import pickle
        f = open(output_path, "wb")
        pickle.dump(self.vocab, f)
        f.close()

    def load(self, input_path):
        import pickle
        f = open(input_path, "rb")
        self.vocab = pickle.load(f)
        self.cnt = len(self.vocab) - 1
