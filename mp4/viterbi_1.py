# mp4.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created Fall 2018: Margaret Fleck, Renxuan Wang, Tiantian Fang, Edward Huang (adapted from a U. Penn assignment)
# Modified Spring 2020: Jialu Li, Guannan Guo, and Kiran Ramnath
# Modified Fall 2020: Amnon Attali, Jatin Arora
# Modified Spring 2021 by Kiran Ramnath (kiranr2@illinois.edu)
from math import log
from math import inf

"""
Part 2: This is the simplest version of viterbi that doesn't do anything special for unseen words
but it should do better than the baseline at words with multiple tags (because now you're using context
to predict the tag).
"""


def viterbi_1(train, test):
    tagsDict = {}
    tagPairsDict = {}
    tagWordPairDict = {}
    tagStartDict = {}
    words = []

    for sentence in train:
        firstTag = sentence[1][1]
        if firstTag in tagStartDict:
            tagStartDict[firstTag] += 1
        else:
            tagStartDict[firstTag] = 1

        for i in range(0, len(sentence)):
            pair = sentence[i]
            prevTag = sentence[i - 1][1]
            word = pair[0]
            tag = pair[1]
            # count words
            if word not in words:
                words.append(word)
            # couint tags
            if tag in tagsDict:
                tagsDict[tag] += 1
            else:
                tagsDict[tag] = 1

            # count tagWord Pair
            if tag in tagWordPairDict:
                if word in tagWordPairDict[tag]:
                    tagWordPairDict[tag][word] += 1
                else:
                    tagWordPairDict[tag][word] = 1
            else:
                tagWordPairDict[tag] = {}
                tagWordPairDict[tag][word] = 1

            # count tag pair
            if i != 0:
                if prevTag in tagPairsDict:
                    if tag in tagPairsDict[prevTag]:
                        tagPairsDict[prevTag][tag] += 1
                    else:
                        tagPairsDict[prevTag][tag] = 1
                else:
                    tagPairsDict[prevTag] = {}
                    tagPairsDict[prevTag][tag] = 1
    i = i + 1

    ####################### LAPALACE SMOOTHING #############
    k = .1
    # calculate initial proba
    number_of_sentences = len(train)
    initial = {}
    for tag in tagStartDict:
        proba = (tagStartDict[tag] + k) / (number_of_sentences + k * len(tagsDict))
        initial[tag] = proba

    # calculate transition  proba
    transition = {}
    for prevTag in tagPairsDict:
        transition[prevTag] = {}
        for nextTag in tagPairsDict[prevTag]:
            proba = (tagPairsDict[prevTag][nextTag] + k) / (tagsDict[prevTag] + k * len(tagsDict))
            transition[prevTag][nextTag] = proba

    # calculate emission proba
    emission = {}

    for tag in tagWordPairDict:
        emission[tag] = {}
        for word in tagWordPairDict[tag]:
            proba = (tagWordPairDict[tag][word] + k) / (tagsDict[tag] + k * (len(words) + 1))
            emission[tag][word] = proba

    # run on test set
    predictions = []
    q = 0
    y = 0

    for sentence in test:
        prediction_i = predict(sentence, initial, transition, emission, tagsDict, words, k)
        predictions.append(prediction_i)

    return predictions


def predict(sentence, initial, transition, emission, tagsDict, words, k):
    prediction = []

    lattice = {}
    backpointer = {}
    firstWord = True
    prevWord = None
    prevTag = None
    index = 0

    for word in sentence:
        lapalace_smoothing = 0
        if firstWord is True:
            lattice[(word, index)] = {}
            backpointer[(word, index)] = {}
            for tag in initial:
                lapalace_smoothing = k / (tagsDict[tag] + k * (len(words) + 1))
                probability = log(initial[tag]) + log(emission[tag].get(word, lapalace_smoothing))
                lattice[(word, index)][tag] = probability
                backpointer[(word, index)][tag] = None
            firstWord = False

        else:
            backpointer[(word, index)] = {}
            lattice[(word, index)] = {}
            for tag in initial:
                lapalace_smoothing = k / (tagsDict[tag] + k * (len(words) + 1))
                emissionProb = emission[tag].get(word, lapalace_smoothing)
                maxProb = -inf
                maxTag = ""
                for prevTag in initial:
                    lapalace_smoothing = k / (tagsDict[prevTag] + k * (len(initial)))
                    currentProb = log(emissionProb) + log(transition[prevTag].get(tag, lapalace_smoothing)) + lattice[
                        (prevWord, index - 1)].get(prevTag, 0)
                    if currentProb > maxProb:
                        maxProb = currentProb
                        maxTag = prevTag
                lattice[(word, index)][tag] = maxProb
                backpointer[(word, index)][tag] = maxTag

        prevWord = word
        index += 1

    i = len(sentence) - 1
    word = sentence[i]
    tag = max(lattice[(word, i)], key=lattice[(word, i)].get)

    while i >= 0:
        word = sentence[i]
        prediction.append((word, tag))
        tag = backpointer[(word, i)][tag]
        i -= 1
    prediction.reverse()

    # print("PREDICTION: ", prediction)

    return prediction