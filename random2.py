# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 07:26:12 2016

@author: priyanka
"""
#==============================================================================
# Microsoft Malware Challenge 
# Input : Command line arguements as described below.
# Outputs file with labels(predictions) of malware families as outputfinal1.txt => str<filename label> 
# Outputs sorted labels in textfile as outputfinal2.txt => int(label)
# Accuracy = 95%
# RunTime = 38 min(approx) on 2 r3.4xlarge nodes with parameters as specified in Readme.
# This is final script which takes 5 command line arguements:<bytePath> <namePath> <nameTestPath> <classPath> <aws_id> <aws_key>
#bytePath = "s3n://eds-uga-csci8360/data/project2/binaries" 
#//s3 path for folder with all .byte files
#namePath = "s3n://eds-uga-csci8360/data/project2/labels/X_train.txt"
#//s3 path for file having file names of training docs
#nameTestPath="s3n://eds-uga-csci8360/data/project2/labels/X_test.txt"
#//s3 path for file having file names of test docs
#classPath = "s3n://eds-uga-csci8360/data/project2/labels/y_train.txt"
#//s3 path for file having labels/classes of training docs
#aws_id = “xxxx”
#//your aws id
#aws_key = “xxxx”
#//your aws key
#==============================================================================

#Imports
from __future__ import division
import urllib
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext, Row
import re
import sys
from pyspark.mllib.feature import HashingTF, IDF
from pyspark.mllib.linalg import SparseVector
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel
from pyspark.mllib.util import MLUtils
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils
from pyspark.sql import SQLContext
from pyspark.ml.feature import NGram

#Function cleans the .byte files before tf calculation occurs.
#Removes the numbers in beginning of each line and /r/n without needing to strip them or use of regex.
def clean(x):
    l= []
    for item in x:
        if (len(item) <= 2):
            l.append(item)     
    return l  
    
def main():
#==============================================================================
# Specify aws credentials
#==============================================================================
    sc = SparkContext(conf = SparkConf().setAppName("Random Forest"))
    sqlContext = SQLContext(sc)
    sc._jsc.hadoopConfiguration().set('fs.s3n.awsAccessKeyId', sys.argv[5])
    sc._jsc.hadoopConfiguration().set('fs.s3n.awsSecretAccessKey', sys.argv[6])
#==============================================================================
# Specify file paths on s3
#==============================================================================

    bytePath = sys.argv[1]
    namePath = sys.argv[2]
    nameTestPath= sys.argv[3]
    classPath = sys.argv[4]

#==============================================================================
# Section 1: GETTING TF OF .BYTE FILES, O/P OF SECTION :(FILE NAME, TF)
#
# Clean all the byte files (removes /r/n and removes the initial number token on each line)
# Generate tf of all the cleaned byte files(includes both training and testing byte files)
#==============================================================================

    #O/P: (FILENAME,TEXT)
    docData= sc.wholeTextFiles(bytePath, 25).map(lambda (x,y):(x.encode("utf-8"),y.encode("utf-8")))
    
    #O/P: (FILENAME, CLEANED TEXT)
    cleanDocData = docData.map(lambda (x,y): (x,clean(y.split())))

    #O/P: Hashing TF arguement -> 256 + 1 (for "??") features
    x = 16**2 + 1
    hashingTF = HashingTF(x) 
    
    #O/P: (FILENAME,TF)
    tfDocData = cleanDocData.map(lambda (x,y): (x,hashingTF.transform(y)))
    
    #cache or persist the output of section 1
    tfDocData.persist()

    print tfDocData.take(1)
#==============================================================================
# Section 2: GETTING LABELS OF TRAINING DATA , O/P OF SECTION:(FILE NAME,LABEL) OF TRAINING DATA
#==============================================================================
    #O/P: (INDEX,FILENAME)
    nameData = sc.textFile(namePath,25).map(lambda x: bytePath+"/"+x+".bytes").zipWithIndex().map(lambda (x,y):(y,x))
    
    #O/P: (INDEX,LABEL) 
    labelData = sc.textFile(classPath,25).zipWithIndex().map(lambda (x,y):(y,str(int(x)-1)))
    
    #O/P: (FILENAME,LABEL)
    joinNameLabel = nameData.join(labelData).map(lambda (x,y):y)
    
    #Cache/persist output of section 2
    joinNameLabel.persist()

#==============================================================================
#  Section 3: Join the tf of byte files with labels for analysis and convert into Labelled Point to feed it into classifier.
#             O/P of section: LabelledPoint(LABEL, TF)
#==============================================================================
    #O/P: (LABEL,TF)
    joinCleanDocLabel = joinNameLabel.join(tfDocData).map(lambda (x,y):y)
    
    #O/P: LabelledPoint(LABEL,TF)
    hashData = joinCleanDocLabel.map(lambda (label,text): LabeledPoint(label,text))
    
    #Persist/cache the output of section 3
    hashData.persist()
  
#==============================================================================
# Section 4: Apply Random Forest Classifier and generate model on hashData using gini impurity.
#            Determined heurestically that 50 trees with depth of 8 give best accuracy.
#==============================================================================
    model = RandomForest.trainClassifier(hashData, numClasses=9, categoricalFeaturesInfo={},
                                      numTrees=50, featureSubsetStrategy="auto",
                                      impurity='gini', maxDepth=8, maxBins=32)
                                
#==============================================================================
# Section 5: Generate test data in the format for prediction. O/P of section: (INDEX,(INDEX,FILENAME,TF))                       
#==============================================================================
                                      
    #O/P: (INDEX,FILENAME)
    nameTestData=sc.textFile(nameTestPath,25).map(lambda x: bytePath+"/"+x+".bytes").zipWithIndex()

    #O/P: (INDEX,FILENAME,TF)
    joinTestDocLabel=nameTestData.join(tfDocData).map(lambda (x,y):(x,y[0],y[1]))
  
    #O/P: (INDEX,(INDEX,FILENAME,TF))
    joinTestDocLabel1=joinTestDocLabel.zipWithIndex().map(lambda (x,y):(y,x))

#==============================================================================
# Section 6: Prediction of Labels and saving the output in a RDD which is saved as text file on s3.
#==============================================================================
    #O/P: Predictions
    prediction = model.predict(joinTestDocLabel1.map(lambda (x,(y,z,w)):w))
    #O/P: (INDEX,FILENAME,PREDICTEDLABEL)
    fullPred = joinTestDocLabel.map(lambda (x,y,z):(x,y)).zip(prediction)

    #O/P: STRING < FILENAME PREDICTEDLABEL>
    fullPred1 = fullPred.map(lambda ((x,y),z):str(x) + " "+ str(z+1))
    
    #Save string<filename predictedlabel> RDD on s3
    fullPred1.saveAsTextFile("s3n://pkey-bucket/outputfinal1.txt")
    
#==============================================================================
# Section 7: Convert the predictions into output file format having only labels as integers in each line.This can be fed to autolab.
#==============================================================================

    #O/P: (FILENAME,LABEL)
    op = fullPred1.map(lambda x: x.split()).map(lambda (x,y):(x,float(y)))
   
    #O/P: (FILENAME, INT(PREDICTEDLABEL))
    op1 = op.map(lambda (x,y):(x,int(y)))

    #O/P: (FILENAME,INDEX)
    xtest = nameTestData.map(lambda (x,y):(x.encode("utf-8"),y))
  
    #O/P: (INDEX,PREDICTED LABEL)
    joinOpXtest = xtest.join(op1).map(lambda (x,y):y)

    #O/P: Sort by index so that predictedlabels are in order.
    j = joinOpXtest.sortByKey()
    #O/P : LABELS
    final = j.map(lambda (x,y):y)
    #Save labels as text file on s3
    final.saveAsTextFile("s3n://pkey-bucket/outputfinal2.txt")
    print "final predictions"
    for x in final.collect():
        print x
    
    print "prog completed"

if __name__ == "__main__":
    main()
