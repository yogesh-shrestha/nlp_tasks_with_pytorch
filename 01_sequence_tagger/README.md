# Sequence Tagger

## Task Description
Task is to implement a simple sequence tagger using word embeddings for named entity recognition. Model Architecture is a bidirectional LSTM with a single 100-dimenstional hidden layer. The word embeddings are pretrained which has the dimension of 50. Macro-averaged F1-Score is the evaluation that has been used to evaluate the performance of the model. The dataset is highly dominated a single label, so accuracy is not an appropriate metric for evaluation.

<b>Hyperparameters:</b></br>
Loss Function: Cross-Entropy Loss</br>
Batch-size: 1</br>
Training Cycles: 20</br>
Optimizer: ADAM</br>
Learning rate: 0.001</br>
Embedding Dimension: 50</br>

## Performance of Model on Validation set
Macro-Averaged F1-Score was used to measure the performance of the model on validation set for every epochs. 

<img src='img/Macro-averaged-F1-Score.png'/>

## Performance of Model of Test Set

Micro-averaged F1-Score: 0.818995

Performance of the model can also be measured using the confusion matrix which gives the overview of true positive, true negative, false postive and false negative counts.

<img src='img/confusion_matrix.png'/>