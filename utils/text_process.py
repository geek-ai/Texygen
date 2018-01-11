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
                continue
                # paras += '\n'
                # break
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
        word_index_dict[word] = str(index)
        index_word_dict[str(index)] = word
        index += 1
    return word_index_dict, index_word_dict


# def text_precess(train_text_loc, test_text_loc=None):
#     train_tokens = get_tokenlized(train_text_loc)
#     if test_text_loc is None:
#         test_tokens = list()
#     else:
#         test_tokens = get_tokenlized(test_text_loc)
#     word_set = get_word_list(train_tokens + test_tokens)
#     # text = train_text + test_text
#     [word_index_dict, index_word_dict] = get_dict(word_set)
#     with open('save/word_index_dict.json', 'w') as outfile:
#         json.dump(word_index_dict, outfile)
#     with open('save/index_word_dict.json', 'w') as outfile:
#         json.dump(index_word_dict, outfile)
#     if test_text_loc is None:
#         sequence_len = len(max(train_tokens, key=len))
#     else:
#         sequence_len = max(len(max(train_tokens, key=len)), len(max(test_tokens, key=len)))
#     with open('save/eval_data.txt', 'w') as outfile:
#         outfile.write(text_to_code(test_tokens, word_index_dict, sequence_len))
#     paras = dict()
#     paras['sequence_len'] = sequence_len
#     paras['vocab_size'] = len(word_index_dict)
#     with open('paras/paras.json', 'w') as outfile:
#         json.dump(paras, outfile)
#     return sequence_len, len(word_index_dict) + 1, 'save/word_index_dict.json', 'save/index_word_dict.json'

def text_precess(train_text_loc, test_text_loc=None):
    train_tokens = get_tokenlized(train_text_loc)
    if test_text_loc is None:
        test_tokens = list()
    else:
        test_tokens = get_tokenlized(test_text_loc)
    word_set = get_word_list(train_tokens + test_tokens)
    [word_index_dict, index_word_dict] = get_dict(word_set)

    if test_text_loc is None:
        sequence_len = len(max(train_tokens, key=len))
    else:
        sequence_len = max(len(max(train_tokens, key=len)), len(max(test_tokens, key=len)))
    with open('save/eval_data.txt', 'w') as outfile:
        outfile.write(text_to_code(test_tokens, word_index_dict, sequence_len))

    return sequence_len, len(word_index_dict) + 1
