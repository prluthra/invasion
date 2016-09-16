# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 20:05:35 2016

@author: priyanka
"""

#final code for normal baseline random forest
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
#==============================================================================
# Specifying paths
#==============================================================================
    sc = SparkContext(conf = SparkConf().setAppName("Random Forest"))
    sqlContext = SQLContext(sc)
#    bytePath = "/Users/priyanka/Desktop/project2files/all"
#    namePath = "/Users/priyanka/Desktop/X_train_small.txt"
#    nameTestPath="/Users/priyanka/Desktop/X_test_small.txt"
#    classPath = "/Users/priyanka/Desktop/y_train_small.txt"
    bytePath = "s3n://eds-uga-csci8360/data/project2/binaries"
    namePath = "s3n://eds-uga-csci8360/data/project2/labels/X_train.txt"
    nameTestPath="s3n://eds-uga-csci8360/data/project2/labels/X_test.txt"
    classPath = "s3n://eds-uga-csci8360/data/project2/labels/y_train.txt"

#==============================================================================
# SECTION 1: GETTING TF OF ALL .BYTE FILES, O/P OF SECTION :(FILE NAME, TF)
#==============================================================================
    #O/P :(FILE NAME,TEXT)
    docData= sc.wholeTextFiles(bytePath, 25).map(lambda (x,y):(x.encode("utf-8"),y.encode("utf-8")))
    print ("docData done")
    docData.take(1)
    #clean docData here - remove 1st word from line and remove /r/n  
    #O/P: (FILE NAME, CLEAN TEXT)
    cleanDocData = docData.map(lambda (x,y): (x,clean(y.split())))

    x = 16**2 +1 
    hashingTF = HashingTF(x) 
    
    #O/P: (FILE NAME, TF)
    tfDocData = cleanDocData.map(lambda (x,y): (x,hashingTF.transform(y)))
    tfDocData.take(1)
#==============================================================================
# Section 2:TRAINING SET: GETTING LABELS , O/P OF SECTION:(FILE NAME,LABEL)
#==============================================================================
    #Output format : (INDEX, FILE NAME)
    nameData = sc.textFile(namePath,25).map(lambda x: "file:"+bytePath+"/"+x+".bytes").zipWithIndex().map(lambda (x,y):(y,x))
    
    #O/P:(INDEX,LABEL)
    labelData = sc.textFile(classPath,25).zipWithIndex().map(lambda (x,y):(y,str(int(x)-1)))
    
    #O/P :(FILE NAME,LABEL)
    joinNameLabel = nameData.join(labelData).map(lambda (x,y):y)
    
#==============================================================================
# Section 3:Get Data and generate Labelled Point O/P: (LABEL,TF)
#==============================================================================
    #O/P:(LABEL,TF)
    joinCleanDocLabel = joinNameLabel.join(tfDocData).map(lambda (x,y):y) 
    
    #O/P: Labelled Point(LABEL,TF)
    hashData = joinCleanDocLabel.map(lambda (label,text): LabeledPoint(label,text))
    
#==============================================================================
# Section 4: Build Classification Model:
#            Applying Random Forest Classification Model on training data
#==============================================================================
    model = RandomForest.trainClassifier(hashData, numClasses=9, categoricalFeaturesInfo={},
                                     numTrees=50, featureSubsetStrategy="auto",
                                     impurity='gini', maxDepth=8, maxBins=32)
                                     

    
#==============================================================================
# Section 5: TEST: GETTING LABELS , O/P OF SECTION:(FILE NAME,INDEX)
#==============================================================================
    nameTestData=sc.textFile(nameTestPath,25).map(lambda x: "file:"+bytePath+"/"+x+".bytes").zipWithIndex()
#==============================================================================
# Section 6: TEST: O/P: (FILE PREDICTEDLABEL) stored as textFile
#==============================================================================
    #O/P:(FILE,INDEX,TF)
    joinTestDocLabel=nameTestData.join(tfDocData).map(lambda (x,y):(x,y[0],y[1]))
    
    #O/P:(INDEX2,(FILE,(INDEX,TF))
    joinTestDocLabel1=joinTestDocLabel.zipWithIndex().map(lambda (x,y):(y,x))
    #Predict on Sparse vector tf
    prediction = model.predict(joinTestDocLabel1.map(lambda (x,(y,z,w)):w))
    
    #Zip Predictions with filename and label, O/P :((FILE, INDEX),predicted label)
    fullPred = joinTestDocLabel.map(lambda (x,y,z):(x,y)).zip(prediction)
    
    #O/P:(file,predicted label) 
    fullPred1 = fullPred.map(lambda ((x,y),z):str(x) + " "+ str(z+1))
    
    #Save (file,pred label) in text file
    fullPred1.saveAsTextFile("output.txt")
    

    
if __name__ == "__main__":
    main()
