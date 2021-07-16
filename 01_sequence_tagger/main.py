from matplotlib import pyplot as plt
import utils
from datasets import NERDataset
from models import BiLSTMTagger
import torch
import torch.nn as nn


def main():
    #===================Prepare data=============================================================
    # load and split the datasets
    train_data = utils.load_split_data("dataset/train.conll")
    dev_data = utils.load_split_data("dataset/dev.conll")
    test_data = utils.load_split_data("dataset/test.conll")
    # build dictinaries of {word: index} and {tag:index}
    word_index, tag_index = utils.build_word_tag_index([train_data, dev_data, test_data])
    # create sequences of words / tags of each dataset
    train_word_seq, train_tag_seq = utils.create_sequence(train_data, word_index, tag_index)
    test_word_seq, test_tag_seq = utils.create_sequence(test_data, word_index, tag_index)
    dev_word_seq, dev_tag_seq = utils.create_sequence(dev_data, word_index, tag_index)
    # create torch.utils.data.Dataset type
    train_dataset = NERDataset(train_word_seq, train_tag_seq)
    test_dataset = NERDataset(test_word_seq, test_tag_seq)
    dev_dataset = NERDataset(dev_word_seq, dev_tag_seq)
    #----- Create Embedding Matrix ---------------------------------------
    #Filepath of pre-trained Embeddings
    FILE_PATH_EMB = "glove.6B.50d.txt"
    # Dimension of pre-tained embedding is 50
    EMB_DIM = 50
    embedding_matrix = torch.from_numpy(utils.create_embedding_matrix(FILE_PATH_EMB, word_index, EMB_DIM))
    #=========== Building and Training Model===========================================================
    # Hyperparameters----------------------------------------------------
    # size of each layer of LSTM and Linear layer
    hidden_size = 100
    # num of layers of LSTM
    n_layers = 1
    # numbers of classes / tags
    n_tags = len(tag_index)
    # model---------------------------------------------------------------
    tagger = BiLSTMTagger(hidden_size, n_layers, n_tags, embedding_matrix)
    # Loss and optimizer------------------------------------------------------
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(tagger.parameters())
    # Training Model------------------------------------------------------------
    N_EPOCHS = 20
    # list variable for dev dataset 
    macro_avg_f1_scores = []
    # start training the model------------------------------------------------
    for epoch in range(N_EPOCHS):   
        for i, (sentence, tags) in enumerate(train_dataset):
            tagger.zero_grad()    
            tag_scores = tagger(sentence)       
            train_loss = loss_function(tag_scores, tags)
            train_loss.backward()
            optimizer.step()              
        # accuracy, loss of train_dataset
        with torch.no_grad():
            sum_loss_per_epoch = 0.0
            # true tags of all senteces
            train_true_tags = torch.tensor([])
            # predicted tags of all senteces
            train_pred_tags = torch.tensor([])
            for sentence, tags in train_dataset:
                train_true_tags = torch.cat((train_true_tags, tags))
                tag_scores = tagger(sentence)
                train_loss = loss_function(tag_scores, tags)
                sum_loss_per_epoch += train_loss
                _, p_tags = torch.max(tag_scores, 1)
                train_pred_tags = torch.cat((train_pred_tags, p_tags))
            avg_train_loss = sum_loss_per_epoch / len(train_dataset)
            n_correct = (train_true_tags == train_pred_tags).sum().item()
            train_accuracy = n_correct / len(train_true_tags)
        # accuracy, loss, Macro F1-score of dev_dataset
        with torch.no_grad(): 
            sum_loss_per_epoch = 0.0
            # true tags of all senteces
            dev_true_tags = torch.tensor([])
            # predicted tags of all senteces
            dev_pred_tags = torch.tensor([])
            for sentence, tags in dev_dataset:
                dev_true_tags = torch.cat((dev_true_tags, tags))
                tag_scores = tagger(sentence)
                val_loss = loss_function(tag_scores, tags)
                sum_loss_per_epoch += val_loss
                _, p_tags = torch.max(tag_scores, 1)
                dev_pred_tags = torch.cat((dev_pred_tags, p_tags))
            avg_val_loss = sum_loss_per_epoch / len(dev_dataset)
            n_correct = (dev_true_tags == dev_pred_tags).sum().item()
            val_accuracy = n_correct / len(dev_true_tags)
            macro_avg_f1_score = utils.macro_f1_score(dev_true_tags, dev_pred_tags, tag_index)
            macro_avg_f1_scores.append(macro_avg_f1_score)
        # print the summary for each epoch
        print(f'Epoch {epoch+1}/{N_EPOCHS},', 
            f'train_loss={avg_train_loss:.4f},',
            f'val_loss={avg_val_loss:.4f},',
            f'train_accuracy={val_accuracy:.4f},',
            f'val_accuracy={train_accuracy:.4f},',
            f'val_macro_avg_f1_score={macro_avg_f1_score:.6f}')
    # End of training model ---------------------------------------------------
    # Evaluation with testdata -------------------------------------------------
    true_test_tags = torch.tensor([])
    pred_test_tags = torch.tensor([])
    with torch.no_grad():
        for sentence, tags in test_dataset:
            true_test_tags = torch.cat((true_test_tags, tags))
            tag_scores = tagger(sentence)
            _, p_tags = torch.max(tag_scores, 1)
            pred_test_tags = torch.cat((pred_test_tags, p_tags))
    test_macro_avg_f1_score = utils.macro_f1_score(true_test_tags, pred_test_tags, tag_index)
    print('Macro-averaged F1-score of final model on test data: ', test_macro_avg_f1_score)
    print('Macro_averaged F1-score on dev dataset over all training epochs', macro_avg_f1_scores)
    # compute the confusion matrix
    confusion_matrix = utils.compute_confusion_matrix(true_test_tags, pred_test_tags, tag_index)
    print('confusion matrix for test dataset: \n', confusion_matrix)
    # plot the confusion matrix
    utils.plot_confusion_matrix(confusion_matrix, tag_index)
    # plot macro-averaged F1-score on valid dataset over all training epochs
    utils.plot_curve(macro_avg_f1_scores, 'Macro-averaged-F1-Score')

if __name__ =='__main__':
    main()