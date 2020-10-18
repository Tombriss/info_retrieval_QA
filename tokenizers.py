from nltk import RegexpTokenizer
import string
import re
import unicodedata

class Tokenizer():

  def __init__(self,nlp = None,lemma=False,double=False):

    self.regtokenizer = RegexpTokenizer(r'''\w'|\w+|[^\w\s]''')
    self.nlp = nlp
    self.lemma = lemma
    self.double = lemma

  def strip_accents(self,text):
    # removes accents
    return ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')

  def tokenize(self,text):
    if not self.lemma:
      text = re.sub(r'[^\w\s]',' ',text) # replaces ponctuation by ' '
      text = re.sub(' +', ' ', text) # removes spaces larger than ' '
      text = self.strip_accents(text) # removes accents
      tokens = self.regtokenizer.tokenize(text) # tokenize (could use .split() but safer like this)
      return([token.lower() for token in tokens if token not in string.punctuation and len(token)>1]) 
    else:
      tokens = self.nlp(text)
      if not self.double:
        return([self.strip_accents(token.lemma_.lower()) for token in tokens if tokens.text not in string.punctuation and len(token.text)>1])
      else:
        a = set([self.strip_accents(token.lemma_.lower()) for token in tokens if tokens.text not in string.punctuation and len(token.text)>1])
        b = set([self.strip_accents(token.text.lower()) for token in tokens if tokens.text not in string.punctuation and len(token.text)>1])
        return(list(a.union(b)))

        


      
    
    # lower, makes sure (once again) there is no ponctuation, takes only tokens with more than one char
