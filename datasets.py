import os
import numpy as np
import json

class Dataset():

  def __init__(self,FQuAD_path):
      
    with open(FQuAD_path) as json_file:
        FQuAD_data = json.load(json_file)

    self.data = [] # list of samples {id_query,query,id_context}
    self.contexts_table = {} # dict id : context (used to save memory)

    self.contexts_ids = []
    self.contexts_list = []

    self.x = []
    self.y_true = []

    self.read_FQuAD_data(FQuAD_data)

    self.n_samples = len(self.x)
    
    self.original_data = FQuAD_data

    assert len(self.contexts_ids) == len(self.contexts_list) 
    assert len(self.contexts_ids) == len(self.contexts_table)
    assert len(self.x) == len(self.y_true)
    assert self.n_samples > 0

  def read_FQuAD_data(self,FQuAD_data):
    
    for text in FQuAD_data['data']:
      title = text['title']
      paragraphs = text['paragraphs']
      for ix,pg in enumerate(paragraphs):
        context = pg['context']
        id_context = str(hash(context))

        if id_context not in self.contexts_table:
          self.contexts_ids.append(id_context)
          self.contexts_list.append(context)


        self.contexts_table[id_context] = context
        qas = pg['qas']
        for q in qas:
          self.data.append({'id':q['id'],'query':q['question'],'id_context':id_context})
          self.x.append(q['question'])
          self.y_true.append(len(self.contexts_list)-1)

  def output_results(self,sorted_scores,topk=5):
      
      output = self.original_data.copy()
      
      for text in output['data']:
          title = text['title']
          paragraphs = text['paragraphs']
          for ix,pg in enumerate(paragraphs):
            context = pg['context']
            id_context = str(hash(context))
            qas = pg['qas']
            for iq,q in enumerate(qas):
                topk_scores = sorted_scores[iq,:topk]
                top_matching_contexts = [self.contexts_list[i] for i in topk_scores]
                q['top_matching_contexts'] = top_matching_contexts
      
      with open(os.path.join('results','pred.json'), 'w') as fp:
        json.dump(output, fp)
    

class Datasets():

  def __init__(self,FQuAD_train,FQuAD_valid):

    self.train = Dataset(FQuAD_train)
    self.valid = Dataset(FQuAD_valid)
    
