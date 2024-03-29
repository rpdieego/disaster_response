import re
import nltk
nltk.download(['punkt', 'stopwords','wordnet'])
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

def tokenize(text):
    '''
    INPUT: text (string to be cleaned and tokenized)

    OUTPUT: clean_tokens (list cleaned tokens)

    The function clean the strings (remove special characteres, numbers, set it all to lower case and strip),
    tokenize, remove stop words, stemm and then lemmatize. All the tokens are stored as a list in clean_tokens;

    '''
    
    #remove special characteres and swap numbers by 'digit'
    text = re.sub(r"[^a-zA-Z0-9]"," ",text)
    text = re.sub(r"[0-9]", "digit", text)

    #tokenize and remove stop words
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stopwords.words("english")]

    # stemming
    stemmed = [SnowballStemmer('english').stem(t) for t in tokens]

    # instantianting the lemmatizer
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens
