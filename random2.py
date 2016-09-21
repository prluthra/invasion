# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 07:26:12 2016

@author: priyanka
"""
#==============================================================================
# Microsoft Malware Challenge 
# Input : Byte files,Training file names, Test file names, Training Labels
# Outputs file with labels(predictions) of malware families as output.txt
# Outputs sorted labels in textfile as finalOutput.txt
#==============================================================================

#Imports
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

#Function cleans the .byte files before tf calculation occurs.
#Removes the numbers in beginning of each line and /r/n without needing to strip them or use of regex.
def clean(x):
    l= []
    for item in x:
        if len(item) <= 2:
            l.append(item)     
    return l  
    
def main():
#==============================================================================
# Specify aws credentials
#==============================================================================
    sc._jsc.hadoopConfiguration().set('fs.s3n.awsAccessKeyId', '')
    sc._jsc.hadoopConfiguration().set('fs.s3n.awsSecretAccessKey', '')
#==============================================================================
# Specify file paths on s3
#==============================================================================
    bytePath = "s3n://eds-uga-csci8360/data/project2/binaries"
    namePath = "s3n://eds-uga-csci8360/data/project2/labels/X_train.txt"
    nameTestPath="s3n://eds-uga-csci8360/data/project2/labels/X_test.txt"
    classPath = "s3n://eds-uga-csci8360/data/project2/labels/y_train.txt"

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
    x = 16**2 +1 
    hashingTF = HashingTF(x) 
    
    #O/P: (FILENAME,TF)
    tfDocData = cleanDocData.map(lambda (x,y): (x,hashingTF.transform(y)))
    
    #cache or persist the output of section 1
    tfDocData.persist()
#==============================================================================
# Section 2: GETTING LABELS OF TRAINING DATA , O/P OF SECTION:(FILE NAME,LABEL) OF TRAINING DATA
#==============================================================================
    #O/P: (INDEX,FILENAME)
    nameData = sc.textFile(namePath,25).map(lambda x: bytePath+"/"+x+".bytes").zipWithIndex().map(lambda (x,y):(y,x))
    
    #O/P: (INDEX,LABEL) 
    labelData = sc.textFile(classPath,25).zipWithIndex().map(lambda (x,y):(y,str(int(x)-1)))
    
    #O/P: (FILENAME,LABEL)
    joinNameLabel = nameData.join(labelData).map(lambda (x,y):y)
    
    #Cache/persist the output of section 2
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
# Section 5: Generate test data in the format for prediction. O/P of section: ()                        
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

    for x in fullPred1.collect():
        print x   
    
    #Save string<filename predictedlabel> RDD on s3
    fullPred1.saveAsTextFile("s3n://pkey-bucket/output/output.txt")  

#==============================================================================
# Section 7: Convert the predictions into output file format having only labels as integers in each line.This can be fed to autolab.
#==============================================================================

    #op = sc.textFile("s3n://pkey-bucket/output/op2.txt").map(lambda x:x.encode("utf-8")).map(lambda x: x.split()).map(lambda (x,y):(x,float(y)))

    #O/P: ()
    op = fullPred1.map(lambda x: x.split()).map(lambda (x,y):(x,float(y)))
    #O/P: (FILENAME, INT(PREDICTEDLABEL))
    op1 = op.map(lambda (x,y):(x,int(y)))
    
    #xtest = sc.textFile(nameTestPath).map(lambda x: bytePath+"/"+x+".bytes").zipWithIndex().map(lambda (x,y):(x.encode("utf-8"),y))
    #O/P: (FILENAME,INDEX)
    xtest = nameTestData.map(lambda (x,y):(x.encode("utf-8"),y))
    #O/P: (INDEX,PREDICTED LABEL)
    joinOpXtest = xtest.join(op1).map(lambda (x,y):y)
    #O/P: Sort by index so that predictedlabels are in order.
    j = joinOpXtest.sortByKey()
    #O/P : LABELS
    final = j.map(lambda (x,y):y)
    #Save labels as text file on s3
    final.saveAsTextFile("s3n://pkey-bucket/output/finalOutput.txt")
    

if __name__ == "__main__":
    main()