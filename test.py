# from utils.oracle.OracleCfg import OracleCfg
#
# oracle = OracleCfg(sequence_len=20)
# oracle.generat_oracle()


from nltk import Nonterminal, nonterminals, Production, CFG
from nltk.parse import RecursiveDescentParser
import nltk
cfg_grammar = """
  S -> S PLUS x | S SUB x |  S PROD x | S DIV x | x
  PLUS -> '+'
  SUB -> '-'
  PROD -> '*'
  DIV -> '/'
  x -> 'x'
"""
grammar = nltk.CFG.fromstring(cfg_grammar)
parser = nltk.ChartParser(grammar)

sentence1 = 'x  x - x'.split()
for tree in parser.parse(sentence1):
    print(tree)

# print(len(parser.parse(sentence1)))

