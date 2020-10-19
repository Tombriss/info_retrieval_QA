# Main file

import argparse
import os
import numpy as np

from datasets import Datasets
from models.freq_model import Freq_Model
from tokenizers import Tokenizer

# Parser for CLI

parser = argparse.ArgumentParser(description='Context ranking model')
parser.add_argument('-t','--data_train_path')
parser.add_argument('-v','--data_valid_path')
parser.add_argument('-r','--retrain', action='store_true')
parser.add_argument('-p','--params_path',default='params.json')

args = parser.parse_args()

# Loading data, tokenizer and the model
    
datasets = Datasets(args.data_train_path,args.data_valid_path)
tokenizer = Tokenizer()
freq_model = Freq_Model(datasets,tokenizer)

# We train again if --retrain has been called

if args.retrain:
    freq_model.fit()
else: # Otherwise, we load the trained parameters
    freq_model.load(args.params_path)

# Prediction 
freq_model.predict(tp='valid')

# Evaluation (to get metrics values)
freq_model.evaluate(tp='valid')