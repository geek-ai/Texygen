import nltk
from utils.metrics.Metrics import Metrics
from nltk import ngrams

class RefUniqueGram(Metrics):
    def __init__(self, test_text='',ref_text='', gram=3):
        super().__init__()
        self.name = 'RefUniqueGram'
        self.test_data = test_text
        self.ref_data=ref_text
        self.gram = gram
        self.sample_size = 500
        self.test_text=None
        self.reference_text = None
        self.is_first = True
    
    def get_score(self, ignore=False):
        if ignore:
            return 0
        if self.is_first:
            self.get_reference()
            self.get_test()
            self.is_first = False
        return self.get_ng()

    def get_ng(self):
        documentRef = self.get_reference()
        documentTest= self.get_test()
        length = len(documentTest) 
        gramsRef = list()
        gramsTest = list()
        for sentence in documentRef:
            gramsRef += self.get_gram(sentence)
        
        for sentence in documentTest:
            gramsTest += self.get_gram(sentence)
        

        return len(set(gramsTest).difference(set(gramsRef)))/length

    def get_gram(self, tokens):
        grams = list()
        if len(tokens) < self.gram:
            return grams
        gram_generator = ngrams(tokens, self.gram)
        for gram in gram_generator:
            grams.append(gram)
        return grams


    def get_reference(self):
        if self.reference_text is None:
            reference = list()
            with open(self.ref_data) as ref_text:
                for text in ref_text:
                    #text = text.strip().split(" ")
                    text= nltk.word_tokenize(text)
                    reference.append(text)
            self.reference_text = reference
            return reference
        else:
            return self.reference_text

    def get_test(self): 
        if self.test_text is None:
            test = list()
            with open(self.test_data) as test_text:
                for text in test_text:
                    text = nltk.word_tokenize(text)
                    test.append(text)
            self.test_text = test
            return test
        else:
            return self.test_text

