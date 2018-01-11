# from utils.oracle.OracleCfg import OracleCfg
#
# oracle = OracleCfg(sequence_len=20)
# oracle.generat_oracle()
import nltk
from nltk.parse.generate import generate

cfg_grammar = """
  S -> S PLUS x | S SUB x |  S PROD x | S DIV x | x | '(' S ')' 
  PLUS -> '+'
  SUB -> '-'
  PROD -> '*'
  DIV -> '/'
  x -> 'x'
"""
grammar = nltk.CFG.fromstring(cfg_grammar)
parser = nltk.ChartParser(grammar)
sentences = generate(grammar, depth=7, n=1000)
for s in sentences:
    print(' '.join(s) )
#
# sentence1 = 'x  x - x'.split()
# for tree in parser.parse(sentence1):
#     print(tree)

# print(len(parser.parse(sentence1)))

