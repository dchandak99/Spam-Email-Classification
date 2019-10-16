#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 12:11:46 2019

@author: dchandak99
"""

'''email classification using k nearest neighbours'''
import random
import pandas as pd
import numpy as np
import math
import string
import operator
#filename = "emails.csv"
#data = pd.read_csv(filename)
##print(data)
#data1 = np.array(data)

def loadDataset(filename, split, trainingSet=[], testSet=[]):
    '''read the file and split the data into training and test'''
    data = pd.read_csv(filename)
    data1 = np.array(data)
    random.shuffle(data1)
    data2 = data1[2000:3500]
    for x in range(len(data2)):
        if random.random() < split:
            trainingSet.append(data2[x])
        else:
            testSet.append(data2[x])

def makeDict(trainingSet):
    '''stores list of list of words of trainingSet along with spam labels'''
    return list(map(lambda x: [word_list(x[0]),x[1]], trainingSet))
    
def word_list(text):
    '''removes punctuation and returns list of words from text'''
    return list(filter(lambda x: notNumber(x), (list(map(lambda x: x.lower() , text.translate(str.maketrans('', '', string.punctuation)).split())))))
    #return unique_words(list(filter(lambda x: notNumber(x),text.translate(str.maketrans('', '', string.punctuation)).split())))

def notNumber(x):
    return not x.isdigit()

def all_words_unique(table):
    '''takes output of makeDict and combines all the words in final list'''
    spam_list=[]
    ham_list = []
    for data in table:
        if data[1] == 1:
            spam_list += data[0]
        else:
            ham_list += data[0]
      # print(len(unique_words(word_list)), x)
    return [list(set(spam_list)), list(set(ham_list))]

def euclideanDistance(instance1, instance2):
	distance = 0
	for x in range(len(instance1)):
		distance += pow((instance1[x] - instance2[x]), 2)
	return math.sqrt(distance)

def frequency_list(spam_list, ham_list, table, spam_cut, ham_cut):
    '''returns the list of scores for all spam and ham words'''
    n = len(table)
    spam_total = np.sum(np.array(table)[:,1])
    ham_total =  n - spam_total
    spam_list_scores = {}
    ham_list_scores = {}
    i=0
    for word in spam_list:
        
        spam_count = 0
        ham_count = 0
        for row in table:
            if row[1] == 1:
                spam_count += row[0].count(word)
            else:
                ham_count += row[0].count(word)
        spam_score = spam_count/spam_total
        ham_score = ham_count/ham_total
        spam_list_scores[word] = spam_score -ham_score
#        print(i,word, spam_count, ham_count)
        print(i,word, spam_score, ham_score)
        i = i+1
    print(spam_list_scores)
    top_spam_frequency = np.array(sorted(spam_list_scores.items(), key=operator.itemgetter(1), reverse=True))[:spam_cut,0]
    print(top_spam_frequency)
    
    i=0
    for word in ham_list:
        
        spam_count = 0
        ham_count = 0
        for row in table:
            if row[1] == 1:
                spam_count += row[0].count(word)
            else:
                ham_count += row[0].count(word)
        spam_score = spam_count/spam_total
        ham_score = ham_count/ham_total
        ham_list_scores[word] = ham_score-spam_score
#        print(i,word, spam_count, ham_count)
        print(i,word, spam_score, ham_score)
        i = i+1
    print(ham_list_scores)
    top_ham_frequency = np.array(sorted(ham_list_scores.items(), key=operator.itemgetter(1), reverse=True))[:ham_cut,0]
    print(top_ham_frequency)
    
    return top_spam_frequency.tolist(), top_ham_frequency.tolist()

def feature_vector(mail_words, top_word_list):
    '''top_word_list is the list of all words from frequency_list'''
    word_vector=[]
    for word in top_word_list:
        if word in mail_words:
            word_vector.append(1)
        else:
            word_vector.append(0)
    return word_vector

def mapped_data(trainingSet, top_word_list):
    '''will map the output from makedict to every mail's feature vector'''
    '''trainingSet = myDict'''
    return list(map(lambda x: [feature_vector(word_list(x[0]), top_word_list),x[1]], trainingSet))


    
def getNeighbors(trainingSet, testMail, k, top_word_list):
    distances = []
    i=0
    a = mapped_data(trainingSet, top_word_list)
    b = feature_vector(word_list(testMail.tolist()[0]), top_word_list)
    for x in range(len(trainingSet)):
        dist = euclideanDistance(b, a[x][0])
#        print(len(trainingSet),i,dist)
        distances.append((trainingSet[x], dist))
        i+=1
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
#        neighbors += [[distances[x][0][1],dist]]
        neighbors.append(distances[x][0])
    return neighbors



#def getResponse(neighbors,threshold):
#	classVotes = {}
#	for x in range(len(neighbors)):
#		response = neighbors[x][-1]
#		if response in classVotes:
#			classVotes[response] += 1
#		else:
#			classVotes[response] = 1
#	sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
#	return sortedVotes[0][0]
def getResponse(neighbors):
	classVotes = {}
	for x in range(len(neighbors)):
		response = neighbors[x][-1]
		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1
	sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]
    

def getAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] == predictions[x]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0

def main():
    trainingSet=[]
    testSet=[]
    split = 0.7 
    loadDataset("emails.csv", split, trainingSet, testSet)
    print('Train set: ' + repr(len(trainingSet)))
    print('Test set: ' + repr(len(testSet)))
    predictions=[]
    table = makeDict(trainingSet)
    #print(table)
    b = all_words_unique(table)
    #print(b)
    spam_list = b[0]
    #print(spam_list)
    ham_list = b[1]
    #print(ham_list)
    top_words = frequency_list(spam_list, ham_list, table,120,40)
    #print(top_words)
    top_word_list = top_words[0] + top_words[1]
    #print(top_word_list)
    k = 5
    correct = 0
    incorrect = 0
    for x in range(len(testSet)):
        neighbors = getNeighbors(trainingSet, testSet[x], k,top_word_list)
        result = getResponse(neighbors)#, 0.2)
        predictions.append(result)
        if result == testSet[x][-1]:
            correct +=1
        else:
            incorrect += 1
        print(len(testSet),x,'> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]),correct, incorrect, (correct/(incorrect+correct))*100)
    accuracy = getAccuracy(testSet, predictions)
    print('Accuracy: ' + repr(accuracy) + '%')
	
main()

#trainingSet=[]
#testSet=[]
#split = 0.67
#loadDataset("emails.csv", split, trainingSet, testSet)
#print('Train set: ' + repr(len(trainingSet)))
#print('Test set: ' + repr(len(testSet)))
#predictions=[]
#table = makeDict(trainingSet)
#print(table)
#b = all_words_unique(table)
#print(b)
#spam_list = b[0]    
#print(spam_list)   
#ham_list = b[1]
#print(ham_list)
#top_words = frequency_list(spam_list, ham_list, table,150,50)
#print(top_words)
#top_word_list = top_words[0] + top_words[1]
#print(top_word_list)
#k = 3
#correct = 0
#incorrect = 0
#for x in range(len(testSet)):
#    neighbors = getNeighbors(trainingSet, testSet[x], k,top_word_list)
#    result = getResponse(neighbors)#,0.4)
#    predictions.append(result)
#    if result == testSet[x][-1]:
#        correct +=1
#    else:
#        incorrect += 1
#    print(len(testSet),x,'> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]),correct, incorrect, (correct/(incorrect+correct))*100)
#accuracy = getAccuracy(testSet, predictions)
##    accuracy = correct/ (correct+incorrect)*100
#print('Accuracy: ' + str((correct/(incorrect+correct))*100) + '%')
#
##for i in range(len(distances)):
#        sum += values[i]/distances[i]
    
    
  
            
        
    
