import argparse
import os
import numpy as np

from datasets import Datasets
from models.freq_model import Freq_Model
from tokenizers import Tokenizer

parser = argparse.ArgumentParser(description='Context ranking model')
parser.add_argument('-t','--data_train_path')
parser.add_argument('-v','--data_valid_path')
parser.add_argument('-r','--retrain', action='store_true')
parser.add_argument('-p','--params_path',default='params.json')

args = parser.parse_args()
    
datasets = Datasets(args.data_train_path,args.data_valid_path)
tokenizer = Tokenizer()
freq_model = Freq_Model(datasets,tokenizer)

if args.retrain:
    freq_model.fit()
else:
    freq_model.load(args.params_path)

freq_model.predict(tp='valid')
freq_model.evaluate(tp='valid')