# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
import numpy as np
import math
from tqdm import tqdm
from collections import Counter
import reader

"""
This is the main entry point for MP1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""




"""
  load_data calls the provided utility to load in the dataset.
  You can modify the default values for stemming and lowercase, to improve performance when
       we haven't passed in specific values for these parameters.
"""
 
def load_data(trainingdir, testdir, stemming=False, lowercase=False, silently=False):
    print(f"Stemming is {stemming}")
    print(f"Lowercase is {lowercase}")
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset_main(trainingdir,testdir,stemming,lowercase,silently)
    return train_set, train_labels, dev_set, dev_labels


def create_word_maps_uni(X, y, max_size=None):
    """
    X: train sets
    y: train labels
    max_size: you can ignore this, we are not using it

    return two dictionaries: pos_vocab, neg_vocab
    pos_vocab:
        In data where labels are 1 
        keys: words 
        values: number of times the word appears
    neg_vocab:
        In data where labels are 0
        keys: words 
        values: number of times the word appears 
    """
    # count = Counter(X[0])
    # print(count)
    # print(len(y))
    # print(len(X),'X')
    pos_vocab = Counter()
    neg_vocab = Counter()
    total_val = np.zeros(2)

    i = 0
    while i < len(y):
        for word in X[i]:
            total_val[y[i]] += 1
            if y[i] == 1:
                pos_vocab.update({word: 1})
            else: 
                neg_vocab.update({word: 1})
        i = i + 1
    
    return dict(pos_vocab), dict(neg_vocab)


def create_word_maps_bi(X, y, max_size=None):
    """
    X: train sets
    y: train labels
    max_size: you can ignore this, we are not using it

    return two dictionaries: pos_vocab, neg_vocab
    pos_vocab:
        In data where labels are 1 
        keys: pairs of words
        values: number of times the word pair appears
    neg_vocab:
        In data where labels are 0
        keys: words 
        values: number of times the word pair appears 
    """
    #print(len(X),'X')
    pos_vocab = Counter()
    neg_vocab = Counter()
    ##TODO:
    i = 0
    while i < len(y):
        j = 0
        while j < len(X[i]) - 1 :
            curr_words = X[i][j] + " " + X[i][j+1]
            if y[i] == 1:
                pos_vocab.update({curr_words:1})
            else: 
                neg_vocab.update({curr_words:1})
            j = j + 1

        i = i + 1
    pos_uni, neg_uni = create_word_maps_uni(X,y)
    pos_vocab.update(pos_uni)
    neg_vocab.update(neg_uni)
    return dict(pos_vocab), dict(neg_vocab)



# Keep this in the provided template
def print_paramter_vals(laplace,pos_prior):
    print(f"Unigram Laplace {laplace}")
    print(f"Positive prior {pos_prior}")


"""
You can modify the default values for the Laplace smoothing parameter and the prior for the positive label.
Notice that we may pass in specific values for these parameters during our testing.
"""

def naiveBayes(train_set, train_labels, dev_set, laplace=.0001, pos_prior=0.8, silently=False):
    '''
    Compute a naive Bayes unigram model from a training set; use it to estimate labels on a dev set.

    Inputs:
    train_set = a list of emails; each email is a list of words
    train_labels = a list of labels, one label per email; each label is 1 or 0
    dev_set = a list of emails
    laplace (scalar float) = the Laplace smoothing parameter to use in estimating unigram probs
    pos_prior (scalar float) = the prior probability of the label==1 class
    silently (binary) = if True, don't print anything during computations 

    Outputs:
    dev_labels = the most probable labels (1 or 0) for every email in the dev set

    '''
    # Keep this in the provided template
    print_paramter_vals(laplace,pos_prior)

    pos_vocab, neg_vocab =  create_word_maps_uni(train_set,train_labels)
    total_val = np.zeros(2)
    i = 0
    while i < len(train_labels):
        for word in train_set[i]:
            total_val[train_labels[i]] += 1
        i += 1

    proba_pos_vocab = {key: ( (val + laplace)/ (total_val[1]  + laplace * (1 + len(pos_vocab))))  for (key, val) in pos_vocab.items()} 
    proba_neg_vocab = {key: ( (val + laplace)/ (total_val[0] + laplace * (1 + len(neg_vocab))))  for (key, val) in neg_vocab.items()} 
    

    unknown_pos = np.log(laplace / (sum(pos_vocab.values()) + laplace * len(pos_vocab)))
    unknown_neg = np.log(laplace / (sum(neg_vocab.values()) + laplace * len(neg_vocab)))

    #print(pos_sum)

    dev_labels = []

    for email in dev_set:
        pos_post = np.log(pos_prior)
        neg_post = np.log(1 - pos_prior)
        
        for word in email:
            if word in proba_pos_vocab:
                pos_post += np.log(proba_pos_vocab[word])
            else:
                pos_post += unknown_pos
            if word in proba_neg_vocab:
                neg_post += np.log(proba_neg_vocab[word])
            else:
                neg_post += unknown_neg

        if pos_post >= neg_post:
            dev_labels.append(1)
        else:
            dev_labels.append(0)


    print(Counter(dev_labels))
    return dev_labels




# Keep this in the provided template
def print_paramter_vals_bigram(unigram_laplace,bigram_laplace,bigram_lambda,pos_prior):
    print(f"Unigram Laplace {unigram_laplace}")
    print(f"Bigram Laplace {bigram_laplace}")
    print(f"Bigram Lambda {bigram_lambda}")
    print(f"Positive prior {pos_prior}")


def bigramBayes(train_set, train_labels, dev_set, unigram_laplace=0.0001, bigram_laplace=0.01, bigram_lambda=.25,pos_prior=0.75,silently=False):
    '''
    Compute a unigram+bigram naive Bayes model; use it to estimate labels on a dev set.

    Inputs:
    train_set = a list of emails; each email is a list of words
    train_labels = a list of labels, one label per email; each label is 1 or 0
    dev_set = a list of emails
    unigram_laplace (scalar float) = the Laplace smoothing parameter to use in estimating unigram probs
    bigram_laplace (scalar float) = the Laplace smoothing parameter to use in estimating bigram probs
    bigram_lambda (scalar float) = interpolation weight for the bigram model
    pos_prior (scalar float) = the prior probability of the label==1 class
    silently (binary) = if True, don't print anything during computations 

    Outputs:
    dev_labels = the most probable labels (1 or 0) for every email in the dev set
    '''

    pos_vocab, neg_vocab = create_word_maps_bi(train_set, train_labels)

    pos_vocab_uni, neg_vocab_uni = create_word_maps_uni(train_set, train_labels)


    print_paramter_vals_bigram(unigram_laplace,bigram_laplace,bigram_lambda,pos_prior)

    total_val_uni = np.zeros(2)
    i = 0
    while i < len(train_labels):
        for word in train_set[i]:
            total_val_uni[train_labels[i]] += 1
        i += 1

    total_val_bi = total_val_uni.copy() - 1
    proba_neg_vocab = {}
    proba_pos_vocab = {}

    for (key, val) in pos_vocab.items():
        if ' ' in key:
            #two words
            proba_pos_vocab[key] = ( (val + bigram_laplace)/ (total_val_bi[1]  + bigram_laplace * (1 + (len(pos_vocab) - len(pos_vocab_uni))))) 
        else: 
            #one word
            proba_pos_vocab[key] = ( (val + unigram_laplace)/ (total_val_uni[1]  + unigram_laplace * (1 + len(pos_vocab_uni)))) 

    for (key, val) in neg_vocab.items():
        if ' ' in key:
            #two words
            proba_neg_vocab[key] = ( (val + bigram_laplace)/ (total_val_bi[1]  + bigram_laplace * (1 + (len(neg_vocab) - len(neg_vocab_uni))))) 
        else: 
            #one word
            proba_neg_vocab[key] = ( (val + unigram_laplace)/ (total_val_uni[1]  + unigram_laplace * (1 + len(neg_vocab_uni)))) 

    unknown_pos_uni = np.log(unigram_laplace / (sum(pos_vocab.values()) + unigram_laplace * len(pos_vocab_uni)))
    unknown_neg_uni = np.log(unigram_laplace / (sum(neg_vocab.values()) + unigram_laplace * len(neg_vocab_uni)))

    unknown_pos_bi = np.log(bigram_laplace / (sum(pos_vocab.values()) + bigram_lambda * (len(pos_vocab) - len(pos_vocab_uni))))
    unknown_neg_bi = np.log(bigram_laplace / (sum(neg_vocab.values()) + bigram_lambda * (len(neg_vocab) - len(neg_vocab_uni))))

    dev_labels = []

    for email in dev_set:
         ################## uni ################
        pos_post_uni = np.log(pos_prior)
        neg_post_uni = np.log(1 - pos_prior)
        for word in email:
            if word in proba_pos_vocab:
                pos_post_uni += np.log(proba_pos_vocab[word])
            else:
                pos_post_uni += unknown_pos_uni
            if word in proba_neg_vocab:
                neg_post_uni += np.log(proba_neg_vocab[word])
            else:
                neg_post_uni += unknown_neg_uni

        ##################### bi ####################
        pos_post_bi = np.log(pos_prior)
        neg_post_bi = np.log(1 - pos_prior)
        j = 0
        while j < len(email) - 1:
            curr_words = email[j] + " " + email[j+1]
            if curr_words in proba_pos_vocab:
                pos_post_bi += np.log(proba_pos_vocab[curr_words])

            else:
                pos_post_bi += unknown_pos_bi

            if curr_words in proba_neg_vocab:
                neg_post_bi += np.log(proba_neg_vocab[curr_words])
            else:
                neg_post_bi += unknown_neg_bi
            j = j + 1
        final_pos_post = (1- bigram_lambda) * pos_post_uni + bigram_lambda * pos_post_bi
        final_neg_post = (1 - bigram_lambda) * neg_post_uni + bigram_lambda * neg_post_bi

        if final_pos_post >= final_neg_post:
            dev_labels.append(1)
        else:
            dev_labels.append(0)

        #print(Counter(dev_labels))

    max_vocab_size = None


    return dev_labels
