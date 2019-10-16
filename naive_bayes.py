#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 17:19:49 2019

@author: ayushg1235
"""
import random
import pandas as pd
import numpy as np
import math
import string
#filename = "emails.csv"
#data = pd.read_csv(filename)
##print(data)
#
#
#data1 = np.array(data)


def loadDataset(filename, split, trainingSet=[], testSet=[]):
    '''read the file and split the data into training and test'''
    data = pd.read_csv(filename)
    data1 = np.array(data)
    
    random.shuffle(data1)
    data2 = data1[2000:4000]
    for x in range(len(data2)):
        if random.random() < split:
            trainingSet.append(data2[x])
        else:
            testSet.append(data2[x])

    
def makeDict(trainingSet):
    '''stores list of list of words of trainingSet along with spam labels'''
    return list(map(lambda x: [word_list_unique(x[0]),x[1]], trainingSet))
    
def word_list_unique(text):
    '''removes punctuation and returns list of words from text'''
    return (list(set(list(filter(lambda x: notNumber(x), (list(map(lambda x: x.lower() , text.translate(str.maketrans('', '', string.punctuation)).split()))))))))
    #return unique_words(list(filter(lambda x: notNumber(x),text.translate(str.maketrans('', '', string.punctuation)).split())))

def notNumber(x):
    return not x.isdigit()

#def flatten(lst):
#    '''flattens a list'''
#    new_lst = []
#    for sub_list in lst:
#        for x in sub_list:
#            new_lst.append(x)
#    return new_lst

def all_words_unique(table):
    '''takes output of makeDict and combines all the words in final list'''
    word_list=[]
    for x in range(len(table)):
        word_list += table[x][0]
       # print(len(unique_words(word_list)), x)
    return list(set(word_list))


    
def prob_table(trainingSet, dictionary_final, dictionary_part):
    table = []
    spam_count = 0
    ham_count = 0
    n = len(trainingSet)
    spam_total = np.sum(np.array(trainingSet)[:,1])
    ham_total =  n - spam_total
    j=0
    for word in dictionary_final:
        for i in range(len(trainingSet)):
            if word in dictionary_part[i][0]:
                if dictionary_part[i][1] == 1:
                    spam_count += 1
                else:
                    ham_count += 1
        print(len(dictionary_final), j, word) 
        j = j+1
        table += [[word,(1+ spam_count)/(2 + spam_total), (1 + ham_count)/(2 + ham_total)]]
#        print(table)
        spam_count = 0
        ham_count = 0
    return table

#random.shuffle(data1)
#dictionary_part = makeDict(data1[:4000])
#dictionary_final = all_words_unique(dictionary_part)
#final_table = prob_table(data1[:4000], dictionary_final, dictionary_part)

def classify(mail, final_table, dictionary_final, trainingSet):
    mail_words = word_list_unique(mail)
   # cleaned_mail = mail_cleaner(mail_words, dictionary)
    word_vector = []
    for word in dictionary_final:
        if word in mail_words:
            word_vector.append(1)
        else:
            word_vector.append(0)
            
    def P_mail_given_spam(word_vector, final_table):
        prod = 1
        counter = 0
        ten_count= 0
        for x in word_vector:
            if x == 0:
                prod *= (1 - final_table[counter][1]) 
                if prod < pow(10,-10):
                    prod *= 100
                    ten_count += 2
                counter += 1
                
            else:  
                prod *= final_table[counter][1] 
                if prod < pow(10,-10):
                    prod *= 100
                    ten_count += 2
                counter += 1
        return (prod, ten_count)
        
    def P_mail_given_ham(word_vector, final_table):
        prod = 1
        counter = 0
        ten_count= 0
        for x in word_vector:
            if x == 0:
                prod *= (1 - final_table[counter][2])
                if prod < pow(10,-10):
                    prod *= 100
                    ten_count += 2
                counter += 1
            else:  
                prod *= final_table[counter][2]
                if prod < pow(10,-10):
                    prod *= 100
                    ten_count += 2
                counter += 1
        return (prod, ten_count)
        
    n = len(trainingSet)
    spam_total = np.sum(np.array(trainingSet)[:,1])
    ham_total =  n - spam_total
    #spam_prob = 1368/5728
    #ham_prob = 4360/5728
    spam_prob = spam_total/n
    ham_prob = ham_total/n
    
    spam_term = P_mail_given_spam(word_vector, final_table)
    spam_number = spam_term[0] * spam_prob
    spam_power = spam_term[1]        
    ham_term = P_mail_given_ham(word_vector, final_table) 
    ham_number = ham_term[0] * ham_prob
    ham_power = ham_term[1]
    
    if spam_power < ham_power:
        numerator = spam_number
        denominator = spam_number + (ham_number * pow(10, spam_power - ham_power))
        return (numerator/denominator) 
    
    else:
        numerator = spam_number * pow(10, ham_power-spam_power)
        denominator = numerator + ham_number
        return (numerator/denominator)
    
    
def accuracy_pred(testSet, start, end, trainingSet, dictionary_final, final_table):
    correct = 0
    incorrect = 0
    i = start
    for j in range(end-start):
        if testSet[i][1] == 1:
            
            if classify(testSet[i][0], final_table, dictionary_final,trainingSet) > 0.8:
                correct += 1
                print("correct prediction", correct, incorrect)
            else :
                incorrect += 1
                print("incorrect prediction", correct, incorrect)
            i += 1
            
        elif testSet[i][1] == 0:
            if classify(testSet[i][0], final_table, dictionary_final,trainingSet) < 0.2:
                correct += 1
                print("correct prediction", correct, incorrect)
            else:
                incorrect += 1
                print("incorrect prediction", correct, incorrect)
            i += 1
    print("accuracy:", correct/(correct+incorrect) *100)       
    return (correct, incorrect, (correct / (end-start)) * 100)

def main():
    trainingSet = []
    testSet = []
    split = 0.80
    loadDataset("emails.csv", split, trainingSet, testSet)
    #print(len(trainingSet))
    #print(len(testSet))
    dictionary_part = makeDict(trainingSet)
    #print(len(dictionary_part))
    dictionary_final = all_words_unique(dictionary_part)
    #print(len(dictionary_final))
    final_table = prob_table(trainingSet, dictionary_final, dictionary_part)
    accuracy_pred(testSet,0,len(testSet) , trainingSet, dictionary_final, final_table)
    
main()
    
    
    
#trainingSet = []
#testSet = []
#split = 0.85
#loadDataset("emails.csv", split, trainingSet, testSet)
#print(len(trainingSet))
#print(len(testSet))
#dictionary_part = makeDict(trainingSet)
#print(len(dictionary_part))
#dictionary_final = all_words_unique(dictionary_part)
#print(len(dictionary_final))
#final_table = prob_table(trainingSet, dictionary_final, dictionary_part)
#accuracy_pred(testSet,0,len(testSet) , trainingSet, dictionary_final, final_table)
#print("accuracy:", correct/)
    
        


    


    
    
    


    



            
            