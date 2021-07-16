import torch
import numpy as np
from collections import defaultdict
import seaborn
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

#===========================================================================
def load_split_data(file_path: str) -> list:
    """ 
    loads data from the file containing datasets and eturns a list of sentences, 
    where each sentence itself is a list containing word/NER-Tag pairs:
    [[[word1, tag1][word2, tag2]...]..........]
    """
    with open(file_path) as file:
        sentences = []
        single_sentence = []
        for line in file:
            # sentences are separated by empty line, the first line is ignored
            if len(line)==0 or line.startswith('-DOCSTART') or line[0]=="\n":
                if len(single_sentence) > 0:
                    sentences.append(single_sentence)
                    single_sentence = []
                continue
            splits = line.split('\t')
            # only the first and the last elements are relevant for the task
            single_sentence.append([splits[0], splits[-1].rstrip('\n')])
        # last line is reached
        if len(single_sentence) > 0:
            sentences.append(single_sentence)
            single_sentence = []
    return sentences
#==============================================================================
def build_word_tag_index(data_sets: list) -> dict:
    """
    Returns two dictionaries:
    {word: index} and {tag: index}
    """
    # build sets of words and tags
    tag_set = set()
    word_set = set()
    # add words to word_set and tags to tag_Set
    for data_set in data_sets:
        for sentence in data_set:
            for word, tag in sentence:
                tag_set.add(tag)
                word_set.add(word.lower())
    
    # map tags to indices
    sorted_tag = sorted(list(tag_set), key=len)
    tag_index = {}
    for tag in sorted_tag:
        tag_index[tag] = len(tag_index)
        
    # map words to indices
    word_index = {}
    # unknown token is assigned to 0 index 
    if len(word_index) == 0:
        word_index['UNKNOWN_TOKEN'] = len(word_index)
    for word in word_set:
        word_index[word] = len(word_index)   
    return word_index, tag_index
#================================================================================
def create_sequence(data: list, 
                    word_index: dict, 
                    tag_index: dict)->list:
    """
    Returns sequence represention of each sentence
    [
    [5   30 ]
    [125 523  10,15]
    ................
    ]
    ! returns list of tensors
    """
    sent_sequences = []
    tag_sequences = []
    for sent in data:
        sent_sequence_ = []
        tag_sequence_ = []
        for word, tag in sent:
            if word.lower() in word_index:
                word_idx = word_index[word.lower()]
            else: 
                word_idx = word_index['UNKNOWN_TOKEN']
            sent_sequence_.append(word_idx)
            tag_sequence_.append(tag_index[tag])            
        sent_sequences.append(torch.tensor(sent_sequence_))
        tag_sequences.append(torch.tensor(tag_sequence_))        
    return sent_sequences, tag_sequences
#=================================================================================
def create_embedding_matrix(file_path: str, 
                            word_index: dict, 
                            embed_dim: int) -> np.array:
    """
    Returns embedding matrix
    """
    # embedding index: {word: embedding vector}
    embedding_index = {}
    with open(file_path, encoding="utf-8") as file:
        for line in file:
            splits = line.strip().split(' ')
            word = splits[0]
            embed_vector = np.array(splits[1:], dtype='float32')
            embedding_index[word] = embed_vector
    # np array(matrix) with all elements zero, each row is an embedding vector        
    embedding_matrix = np.zeros((len(word_index), embed_dim), dtype=np.float32)
    # substitute zero vectors(rows) with embedding vector
    for word, idx in word_index.items():
        embedding_vector = embedding_index.get(word)
        # embedding vector of unkown tokens is zero vector
        if embedding_vector is not None:
            embedding_matrix[idx] = embedding_vector      
    return embedding_matrix
#==========================================================================================
def macro_f1_score(true_tags: torch.tensor, 
                    pred_tags: torch.tensor, 
                    tag_index:dict)->float:
    """
    # true_tags: list of lists containing true sentence tags
    # pred_tags: list of lists containing predicted sentence tags
    # Returns macro f1 score 
    """
    # convert tensors to lists
    true_tags = true_tags.tolist()
    pred_tags = pred_tags.tolist() 
    # Use negligibly small value to avoid division by zero
    epsilon = 1e-8    
    tags = tag_index.values()
    # TP counts of all tags
    TP = defaultdict(int)
    # FP counts of all tags
    FP = defaultdict(int)
    # FN counts of all tags
    FN = defaultdict(int)
    # count TP, FP, FN
    for true_tag, pred_tag in zip(true_tags, pred_tags):
            if pred_tag == true_tag:
                TP[pred_tag] += 1
            else:
                FP[pred_tag] += 1
                FN[true_tag] += 1                
    # precision: TP /  (TP + FP)
    precision = defaultdict(float)
    # recall: TP / (TP + FN)
    recall = defaultdict(float)   
    # compute precision and recall for all tags
    for tag in tags:
        precision[tag] = TP[tag] / (TP[tag] + FP[tag] + epsilon)
        recall[tag] = TP[tag] / (TP[tag] + FN[tag] + epsilon)    
    # F1-score: 2*precision*recall/(precision + recall)
    f1_score = defaultdict(float)
    # compute F1-Score for all tags
    for tag in tags:
        f1_score[tag] = 2 * precision[tag] * recall[tag] / (precision[tag] + recall[tag] + epsilon)        
    # compute macro-averaged F1-score
    return sum(f1_score.values()) / len(tags)   
#==================================================================================
def compute_confusion_matrix(true_tags: torch.tensor, 
                             pred_tags: torch.tensor, 
                             tag_index:dict)->np.array:
    """
    Computes the confusion matrix with dimension: [num_tags, num_tags]
        - vertical axis represents true condition
        - horizontal axis represents predicted condition
    """
    # square matrix with dim [# tags, # tags]
    confusion_matrix = np.zeros((len(tag_index), len(tag_index)), dtype=np.int32) 
    for t, p in zip(true_tags, pred_tags):
        confusion_matrix[t.int(), p.int()] += 1
    return confusion_matrix
#===================================================================================
def plot_confusion_matrix(confusion_matrix: np.array, tag_index: dict)->None:
    """
    plots the confusion matrix 
    """
    seaborn.set(color_codes=True)
    plt.figure(1, figsize=(15, 10))
    plt.title("Confusion Matrix")
    labels = tag_index.keys()
    ax = seaborn.heatmap(confusion_matrix, 
                    annot=True, 
                    cmap="viridis", 
                    vmin=0, 
                    vmax=1000,
                    xticklabels=labels, 
                    yticklabels=labels
                    )
    ax.set(ylabel="True Label", xlabel="Predicted Label")
    plt.savefig('confusion_matrix.png')
    plt.show()
    plt.close()
#=====================================================================================
def plot_curve(y_values: list, ylabel: str)-> None:
    """
    plots a curve
    """
    x_values = range(1, len(y_values)+1)
    plt.figure(figsize=(12,8))
    plt.plot(x_values, y_values)
    plt.xlabel('Epochs')
    plt.ylabel(ylabel)
    plt.grid()
    plt.savefig(ylabel+'.png')
    plt.show()
    plt.close()
#====================================================================================    