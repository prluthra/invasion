# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 23:03:09 2016

@author: priyanka
"""

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
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils
from pyspark.sql import SQLContext

from pyspark.ml.feature import NGram

def clean(x):
    l= []
    for item in x:
        if len(item) <= 2:
            l.append(item)
    return l  
    

def main():
    sc = SparkContext(conf = SparkConf().setAppName("Random Forest"))
    sqlContext = SQLContext(sc)
    bytePath = "/Users/priyanka/Desktop/project2files/train"
    byteTestPath = "/Users/priyanka/Desktop/project2files/test"
    namePath = "/Users/priyanka/Desktop/X_train_small.txt"
    nameTestPath="/Users/priyanka/Desktop/X_test_small.txt"
    classPath = "/Users/priyanka/Desktop/y_train_small.txt"
    classTestPath = "/Users/priyanka/Desktop/y_test_small.txt"
    
    #docData Output: ('file:/Users/priyanka/Desktop/project2files/train/04mcPSei852tgIKUwTJr.bytes', '00401000 20 FF 58 C0 20 FE 5C 01 F8 00 0F 8B 50 FC 06 01\r\n00401010 8C 01 FF")
    docData= sc.wholeTextFiles(bytePath, 25).map(lambda (x,y):(x.encode("utf-8"),y.encode("utf-8")))
    print ("docData frankie")
    docData.take(1)
    
    
    #clean docData here - remove 1st word from line and remove /r/n  
    cleanDocData = docData.map(lambda (x,y): (x,clean(y.split())))
    
#    #Extract bigrams
#    print "Bigram extract priyanka"
#    dfCleanDocData = sqlContext.createDataFrame(cleanDocData, ["label", "words"])
#    
#    
#    ngram = NGram(inputCol="words", outputCol="ngrams")
#    ngramDataFrame = ngram.transform(dfCleanDocData)
#    
#    for ngrams_label in ngramDataFrame.select("ngrams", "label").take(3):
#        print(ngrams_label)
    
    #try calculating tf here
    x = 16**2 +1 
    hashingTF = HashingTF(x) 
    tfDocData = cleanDocData.map(lambda (x,y): (x,hashingTF.transform(y)))
    tfDocData.take(1)
    #Output format : (0, u'file:/Users/priyanka/Desktop/project2files/train/c2hn9edSNJKmw0OukrBv.bytes')
    nameData = sc.textFile(namePath,25).map(lambda x: "file:"+bytePath+"/"+x+".bytes").zipWithIndex().map(lambda (x,y):(y,x))
    #nameData.take(5)   
    
    #Output format: [(0, '2'), (1, '3'), (2, '2'), (3, '6'), (4, '2')]
    labelData = sc.textFile(classPath,25).zipWithIndex().map(lambda (x,y):(y,str(int(x)-1)))
    
    #Output format: (u'file:/Users/priyanka/Desktop/project2files/train/c2hn9edSNJKmw0OukrBv.bytes', '2'),
    joinNameLabel = nameData.join(labelData).map(lambda (x,y):y)
    #joinNameLabel.take(5) 
    joinCleanDocLabel = joinNameLabel.join(tfDocData).map(lambda (x,y):y) 
    hashData = joinCleanDocLabel.map(lambda (label, text): LabeledPoint(label,text))
    print "hashing TF done"
    #    model = NaiveBayes.train(hashData)
    
    print ("generating model fliss")
    model1 = RandomForest.trainClassifier(hashData, numClasses=9, categoricalFeaturesInfo={},
                                         numTrees=50, featureSubsetStrategy="auto",
                                         impurity='gini', maxDepth=8, maxBins=32)
    #error: 31.36
    #==============================================================================
    # Testing starts here
    #==============================================================================
    docTestData = sc.wholeTextFiles(byteTestPath, 25).map(lambda (x,y):(x.encode("utf-8"),y.encode("utf-8")))
    #docTestData.take(1)
    
    cleanDocTestData = docTestData.map(lambda (x,y): (x,clean(y.split())))
    tfDocTestData = cleanDocTestData.map(lambda (x,y): (x,hashingTF.transform(y)))
    
    nameTestData=sc.textFile(nameTestPath,25).map(lambda x: "file:"+byteTestPath+"/"+x+".bytes").zipWithIndex().map(lambda (x,y):(y,x))
    labelTestData = sc.textFile(classTestPath,25).zipWithIndex().map(lambda (x,y):(y,str(int(x)-1)))
    joinTestNameLabel=nameTestData.join(labelTestData).map(lambda (x,y):y)
    
    joinTestDocLabel=joinTestNameLabel.join(tfDocTestData).map(lambda (x,y):y)
    
    print ("hashing test kenny")
    hashTestData = joinTestDocLabel.map(lambda (label, text): LabeledPoint(label, text))
    hashTestData.persist()
    
    #Random forest prediction and labels and accuracy
    print "prediction part lyndz"
    prediction1 = model1.predict(hashTestData.map(lambda x: x.features))
    labelsAndPredictions1 = hashTestData.map(lambda lp: lp.label).zip(prediction1)
    testErr1 = labelsAndPredictions1.filter(lambda (v, p): v != p).count() / float(hashTestData.count())
    print('Test Error = ' + str(testErr1))
    print('Learned classification forest model:')
    print(model1.toDebugString())
    
    # Save and load random forest model
    model1.save(sc, "/Users/priyanka/Desktop/project2files/myRandomForestClassificationModel1")
    sameModel1 = RandomForestModel.load(sc, "/Users/priyanka/Desktop/project2files/myRandomForestClassificationModel1")

if __name__ == "__main__":
    main()

