import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rankdata
import os 

class IR_Model():

  """
  Abstract class (at least used as) for information retrieval models : 
  encapsulates the common points each information retrieval model we build should have
  """

  def __init__(self,datasets):

    self.datasets = datasets

    self.y_pred_train = []
    self.y_pred_valid = []

    self.y_rank_train = []
    self.y_rank_valid = []

    self.scores_train = []
    self.scores_valid = []
  
  def fit(self):
    raise NotImplementedError
  
  # Routing of attributes according to training or validation
  def predict(self,tp='valid'):

    data = self.datasets.valid if tp == 'valid' else self.datasets.train
    y_pred = self.y_pred_valid if tp == 'valid' else self.y_pred_train
    y_rank = self.y_rank_valid if tp == 'valid' else self.y_rank_train
    scores = self.scores_valid if tp == 'valid' else self.scores_train

    y_pred.clear()
    y_rank.clear()
    scores.clear()

    self._predict(data,y_pred,y_rank,scores)
    
    sorted_scores = (-np.vstack(scores)).argsort(axis = 1)
    
    data.output_results(sorted_scores)
  
  # Will be implemented in child classes
  def _predict(self,data,y_pred,y_rank,scores):
    raise NotImplementedError
  
  # Routing of attributes according to training or validation
  def evaluate(self,tp='valid'):

      y_pred = self.y_pred_valid if tp == 'valid' else self.y_pred_train
      y_rank = self.y_rank_valid if tp == 'valid' else self.y_rank_train
      data = self.datasets.valid if tp == 'valid' else self.datasets.train
      scores = self.scores_valid if tp == 'valid' else self.scores_train

      y_rank.clear()

      return(self._evaluate(y_pred,y_rank,scores,data))

  # Model evaluation
  def _evaluate(self,y_pred,y_rank,scores,data):

    for i in range(len(y_pred)):
      rank = len(data.contexts_list) - rankdata(scores[i])[data.y_true[i]]
      y_rank.append(rank)

    min_range, max_range = 0, 30
    min_all, max_all = min(y_rank), max(y_rank)
    range_ratio = (max_all - min_all) / (max_range - min_range)
    p = plt.hist(y_rank, bins=int(round(100 * range_ratio)),cumulative=True, density=True)
    plt.xlim(min_range, max_range)
    
    if not os.path.exists('results'):
        os.makedirs('results')
    
    plt.savefig(os.path.join('results','ranks_hist.png'))
    print("\naccuracy@1 = {:.2%}".format(np.equal(y_pred,data.y_true).sum()/data.n_samples))
    print("accuracy@5 = {:.2%}".format(p[0][5]))
    print("accuracy@30 = {:.2%}".format(p[0][30]))
    print("mean rank : {0:.2f}".format(np.mean(y_rank)))
    print("median rank : {0:.2f}".format(np.median(y_rank)))
    