# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 11:08:12 2016

@author: priyanka
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 23:23:09 2016

@author: priyanka
"""

#Project 2 - prepare data files and raw tf idf and naive bayes on byte file

from __future__ import division
import urllib
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext, Row
import re
from pyspark.mllib.feature import HashingTF, IDF

from pyspark.mllib.linalg import SparseVector
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel
from pyspark.mllib.util import MLUtils

def read_data_from_url(url):
    return urllib.urlopen(url)

def read_document_data(url):
    #read the entire file
    documentText = read_data_from_url(url)
    #create an array of documents
    data = []
    for line in documentText:
        data.append(line)
    return data 

def clean(doc):
    l = []
    for item in doc:
        item = item.rstrip('\n')
        item = item.rstrip('\r')
        item = item.split(" ",1)[1]
        l.append(item)
    return l

def main():

    sc = SparkContext(conf = SparkConf().setAppName("Prepare data files"))
    
    docLabels = sc.textFile("s3n://eds-uga-csci8360/data/project2/labels/X_train_small.txt")
 
    docLabels.count()
    
    #docLabels.take(5)
    url = docLabels.map(lambda x: "https://s3.amazonaws.com/eds-uga-csci8360/data/project2/binaries/" + x +".bytes")
    
    #data of documents loaded in docData -> [doc1],[doc2]
    docData = url.map(lambda site: read_document_data(site))
    
    #docData.take(1)
    
    print type(docData)
    
    #removed /n,/r and first word in clean function
    docToken= docData.map(lambda x: clean(x))
    
    #docToken.take(1)
    
    #convert doc into bag of words (tokens)
    docStr = docToken.map(lambda x: " ".join(x).split()).zipWithIndex().map(lambda x:(x[1],x[0]))
    
    #docStr.take(1)
    
    classes = sc.textFile("s3n://eds-uga-csci8360/data/project2/labels/y_train_small.txt").map(lambda doc:doc.encode("utf-8")).zipWithIndex().map(lambda x: (x[1],x[0]))
    
    
    joinDocstrClasses = classes.join(docStr).map(lambda x: x[1])
    
    
    hashingTF = HashingTF(50000)
    
    hashData = joinDocstrClasses.map(lambda (label, text): LabeledPoint(label, hashingTF.transform(text)))
    hashData.persist()
    print "hashing TF done"
    
    
#==============================================================================
# Naive bayes    
#==============================================================================
    model = NaiveBayes.train(hashData)
    
    testData = sc.textFile("s3n://eds-uga-csci8360/data/project2/labels/X_test_small.txt",35)
   
    testUrl =  testData.map(lambda x: "/Users/priyanka/Desktop/project2files/test/" + x +".bytes")
    
  
    
    testD = testUrl.map(lambda site: read_document_data(site))
    

    
    testToken =testD.map(lambda x: clean(x))
    
    testStr = testToken.map(lambda x: " ".join(x).split()).zipWithIndex().map(lambda x:(x[1],x[0]))
    

    
    testClasses = sc.textFile("s3n://eds-uga-csci8360/data/project2/labels/y_test_small.txt").map(lambda doc:doc.encode("utf-8")).zipWithIndex().map(lambda x: (x[1],x[0]))
    
    
    
    joinTest = testClasses.join(testStr).map(lambda x: x[1])
   
    
    testHashData = joinTest.map(lambda (label, text): LabeledPoint(label, hashingTF.transform(text)))
    
    prediction_and_labels = testHashData.map(lambda point: (model.predict(point.features), point.label))
    
    correct = prediction_and_labels.filter(lambda (predicted, actual): predicted == actual)
    
    accuracy = correct.count() / float(testHashData.count())
    
    print "Classifier correctly predicted category " + str(accuracy * 100) + " percent of the time"
    
    


if __name__ == "__main__":
    main()