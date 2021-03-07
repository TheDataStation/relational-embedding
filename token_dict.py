import pickle

class TokenDict:
    def __init__(self):
        # token -> numerical id: string -> int
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
    
    def query(self, num):
        for key in iter(self.vocab):
            if self.vocab[key] == num: return key
        return -1
    
    def query_list(self, lst):
        return [self.query(x) for x in lst]
    
    def check_list(self, lst):
        return [self.check(x) for x in lst]
    
    def save(self, output_path):
        f = open(output_path, "wb")
        pickle.dump(self.vocab, f)
        f.close()

    def load(self, input_path):
        f = open(input_path, "rb")
        self.vocab = pickle.load(f)
        self.cnt = len(self.vocab) - 1
