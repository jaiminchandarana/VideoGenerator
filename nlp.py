import nltk
#nltk.download('punkt_tab')
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

if __name__ == "__main__":
    sentence = "Top 5 facts about moon."
    print(tokenize(sentence))