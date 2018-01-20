# # from utils.oracle.OracleCfg import OracleCfg
# #
# # oracle = OracleCfg(sequence_len=20)
# # oracle.generat_oracle()
# import nltk
# from nltk.parse.generate import generate
#
# cfg_grammar = """
#   S -> S PLUS x | S SUB x |  S PROD x | S DIV x | x | '(' S ')'
#   PLUS -> '+'
#   SUB -> '-'
#   PROD -> '*'
#   DIV -> '/'
#   x -> 'x'
# """
# grammar = nltk.CFG.fromstring(cfg_grammar)
# parser = nltk.ChartParser(grammar)
# sentences = generate(grammar, depth=7, n=1000)
# for s in sentences:
#     print(' '.join(s) )
# #
# # sentence1 = 'x  x - x'.split()
# # for tree in parser.parse(sentence1):
# #     print(tree)
#
# # print(len(parser.parse(sentence1)))

# =======================================================================================
# import nltk.corpus
#
# print(str(nltk.corpus.treebank).replace('\\\\','/'))
#
# from nltk.corpus import treebank
# print(treebank.fileids())
#
# print(treebank.words('wsj_0005.mrg'))
#
# from nltk.corpus import ptb
# # nltk.download()
# print(ptb.fileids())
# # nltk.download('all', halt_on_error=False)
# # print(ptb.words('WSJ/00/WSJ_0003.MRG'))
# =======================================================================================

# cfg_grammar = """
#     S -> NP VP
#     VP -> V NP | V NP PP
#     PP -> P NP
#     V -> "saw" | "ate" | "walked"
#     NP -> "John" | "Mary" | "Bob" | Det N | Det N PP
#     Det -> "a" | "an" | "the" | "my"
#     N -> "man" | "dog" | "cat" | "telescope" | "park"
#     P -> "in" | "on" | "by" | "with"
# """

# cfg_grammar = """
#     S -> NP [0.5]
#     S -> VP [0.5]
#     NP -> NP NP [0.5]
#     NP -> 'a' [0.25]
#     NP -> 'e' [0.25]
#     VP -> VP NP [0.9]
#     VP -> 'b' [0.05]
#     VP -> 'c' [0.05]
#
# """
# grammar = PCFG.fromstring(cfg_grammar)
# sentence = generate(grammar, depth=7, n = 100)
# for s in sentence:
#     print(''.join(s))

# print(len(list(generate(grammar,depth=6))))
# =======================================================================================

import numpy as np

# a = np.array([[[ 1,  2,  3],
#   [ 4,  5,  6]],
#  [[ 7,  8,  9],
#   [10, 11, 12]]])
#
#
# b = np.array([[[13, 14],
#   [15, 16],
#   [17, 18]],
#  [[19, 20],
#   [21, 22],
#   [23, 24]]])
a = np.random.randint(19, size=[6, 2, 3])
b = np.random.randint(20, size=[3, 3])
b = np.expand_dims(b, axis=2)
print(b.shape)

# c = (np.matmul(a, b)[1] == np.matmul(a[1], b))
# print(c)
