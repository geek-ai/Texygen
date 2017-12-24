import nltk

from utils.metrics.Metrics import Metrics


class Cfg(Metrics):
    def __init__(self, cfg_grammar=None, test_file=None):
        super().__init__()
        self.name = 'cfg'
        if cfg_grammar is None:
            cfg_grammar = """
              S -> S PLUS x | S SUB x |  S PROD x | S DIV x | x
              PLUS -> '+'
              SUB -> '-'
              PROD -> '*'
              DIV -> '/'
              x -> 'x'
            """
        self.grammar = nltk.CFG.fromstring(cfg_grammar)
        self.parser = nltk.ChartParser(self.grammar)
        self.test_file = test_file

    def get_score(self):
        total_num = 0
        valid_num = 0
        # fixme bad taste
        with open(self.test_file, 'r') as file:
            for s in file:
                s = s.strip('\n')
                if s == '' or s == '\n':
                    continue
                else:
                    total_num += 1
                    for _ in self.parser.parse(s):
                        valid_num += 1
                        break
        return valid_num / total_num
