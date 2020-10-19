from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import numpy as np
from scipy.stats import rankdata
import json

from .ir_model import IR_Model

class Freq_Model(IR_Model):
  
  """
  Class Freq_Model : a model based on tfidf to predict the best context for IR model
  """
  
  def __init__(self,datasets,tokenizer):

    IR_Model.__init__(self,datasets)

    self.tokenizer = tokenizer
    self.vectorizer = TfidfVectorizer(tokenizer = tokenizer.tokenize,max_df=0.12,min_df=1,sublinear_tf=True)
    self.voc_queries = {}
  
  # To fit the training set
  def fit(self):

    print("fitting training data...")

    # Nearly unsupervised : there is only the vocabulary of queries to fit 
    # (and is not necessarly needed)

    for query in tqdm(self.datasets.train.x):
      words_query = set()
      for word in self.tokenizer.tokenize(query):
        if word in words_query:
          continue
        else:
          words_query.add(word)
        if word in self.voc_queries:
          self.voc_queries[word]+= 1/self.datasets.train.n_samples
        else:
          self.voc_queries[word] = 0
          
    with open('params.json', 'w') as fp:
        json.dump(self.voc_queries, fp)

  # To load parameters of a previously trained model
  def load(self,path_params):
      
      with open(path_params, 'r') as fp:
        self.voc_queries = json.load(fp)
    
  # To load parameters of a previously trained model
  def _predict(self,data,y_pred,y_rank,scores):

    print("infering on validation data...")

    tf_idf_ctxts = self.vectorizer.fit_transform(data.contexts_list)
    voc = self.vectorizer.get_feature_names()

    for ix,query in enumerate(tqdm(data.x)):

      tokens_query = self.tokenizer.tokenize(query)
      selected_tokens = [token for token in tokens_query if (token in voc) and (self.voc_queries.get(token,0) < 0.015) ]
      ix_wrds_query = list(set([voc.index(word) for word in selected_tokens]))
      words_ctxts = tf_idf_ctxts[:,ix_wrds_query]
      with np.errstate(divide='ignore'):
        logs = np.log(1+words_ctxts.toarray())
        score_array = np.exp(logs.sum(axis=1))

      ix_pred = np.argmax(score_array)
      y_pred.append(ix_pred)
      scores.append(score_array)

      rank = len(data.contexts_list) - rankdata(score_array)[data.y_true[ix]]