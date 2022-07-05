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
# Modified Spring 2021 by Kiran Ramnath
"""
Part 1: Simple baseline that only uses word statistics to predict tags
"""

def baseline(train, test):
    '''
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    masterDict = {}
    allTags = {}
    #iterate through each sentence 
    for sentence in train:
            #iterate through each word in each senetence 
                for tupleWT in sentence:
                        word = tupleWT[0]
                        tag = tupleWT[1]
                        #add to allTags 
                        if tag in allTags:
                                allTags[tag] += 1
                        else:
                                allTags[tag] = 1


                        if word in masterDict:
                                currDict = masterDict[word]
                                if tag in currDict:
                                        currDict[tag] = currDict[tag] + 1
                                else:
                                        currDict[tag] = 1
                        else:
                                newDict = {}
                                newDict[tag] = 1
                                masterDict[word] = newDict
        
    most_used_tag = max(allTags, key=allTags.get)
    #trained now work with test data
    returnList = []
    print(test[0])
    for sentence in test:
                newSentence = []
                for word in sentence:
                        tag = most_used_tag
                        if word in masterDict:
                                currDictWord = masterDict[word]
                                tag = max(currDictWord, key=currDictWord.get)
                        tupleAdd = (word, tag)
                        newSentence.append(tupleAdd)

                returnList.append(newSentence)


            

    
    return returnList