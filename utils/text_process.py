import nltk


def text_to_code(tokens, dictionary, seq_len):
    code_str = ""
    eof_code = len(dictionary)
    for sentence in tokens:
        index = 0
        for word in sentence:
            code_str += (str(dictionary[word]) + ' ')
            index += 1
        while index < seq_len:
            code_str += (str(eof_code) + ' ')
            index += 1
        code_str += '\n'
    return code_str


def code_to_text(codes, dictionary, seq_len=None):
    paras = ""
    eof_code = len(dictionary)
    # sentence = codes.split('\n')
    for line in codes:
        # numbers = [int(s) for s in line.split() if s.isdigit()]
        numbers = map(int, line)
        for number in numbers:
            if number == eof_code:
                paras += '\n'
                break
            # paras += (dictionary[str(number)] + ' ')
            paras += (dictionary[str(number)] + ' ')
        paras += '\n'
    return paras


def get_tokenlized(file):
    tokenlized = list()
    with open(file) as raw:
        for text in raw:
            text = nltk.word_tokenize(text.lower())
            tokenlized.append(text)
    return tokenlized


def get_word_list(tokens):
    word_set = list()
    for sentence in tokens:
        for word in sentence:
            word_set.append(word)
    return list(set(word_set))


def get_dict(word_set):
    word_index_dict = dict()
    index_word_dict = dict()
    index = 0
    for word in word_set:
        word_index_dict[word] = index
        index_word_dict[index] = word
        index += 1
    return word_index_dict, index_word_dict
