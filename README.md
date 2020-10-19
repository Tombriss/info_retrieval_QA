# Information Retrieval for Question Answering 

## Problematic 

In Question Answering, we are given text and a question refering to the text. The goal is to find the answer to the question in the text. We can find pretty good algorithms to achieve this task. However, they are performing poorly on long texts with multiple paragraphs. A solution to overcome this issue would be to use an other algorithm which would select the potiential paragraph most likely to contain the answer, and finally run the QA algorithm on it. This task should be quite fast for real life applications (several seconds maximum). Furthermore, if the pipeline is fast enough, we can imagine multiple proposals on which we would run the QA algorithm and keep the one with the best confidence. 

## Metrics

A good metric would take into account not only the best match but the top k matching contexts. The average/median rank given to the ground truth contexts could be good ways to quantify the performances. They are good indicators but we have to realize that we don't give the same importance to low or high rank contexts (taking into account wether the ground truth has been classified as 2000th or 2500th is not really important) and both of them depend too much on the number of possible contexts.

No metric is perfect, but the accuracy@k (ratio of questions whose ground truth belongs to the top k matching contexts) for k=5 for example corrects a part of these issues. Moreover, the real life interpretation is straight forward : 70% of accuracy@5 means that for a given question, we have a 70% chance that the ground truth context is among the top 5 matchs. We could then run the QA algorithm on each of the 5 contexts and select the one with the best confidence. For a perfect QA algorithm, this would imply a 70% accuracy on the whole pipeline. 

More generally, we can observe the cumulative histogram of ranks to diagnose our model, but for the model selection, we will keep the accuracy@5.

## Results

For the moment, one of the simplest model seems to be the best. *Freq_model* uses a slightly modified version of tfidf to give a similarity score between a question and a context. 

`accuracy@1 = 64.59%` (exact match) \
`accuracy@5 = 75.31%` \
`accuracy@30 = 89.93%` \
`mean rank = 10.44` \
`median rank = 0.00` 

## Make it run

Requirements :

* `numpy 1.19.1`
* `scikit-learn 0.23.2`
* `tqdm 4.50.2`
* `matplotlib 3.3.1`
* `nltk 3.5`

To infer on validation after retraining the model :

`python context_ranking.py --data_train_path train.json --data_valid_path valid.json --retrain`

To infer on validation without retraining the model :

`python context_ranking.py --data_train_path train.json --data_valid_path valid.json --params_path params.json`

(replacing all the paths with yours)

The first one creates the `params.json` and stores the parameters after training.
The second one loads `params.json` instead of fitting training data.

Both will prompt the metric values, create a `results` directory and put in there `pred.json` to store prediction and the histogram of gt's ranks (`rank_hist.png`).
