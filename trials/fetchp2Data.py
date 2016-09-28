i# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 12:34:26 2016

@author: priyanka
"""
#Automating script to fetch .byte and .asm files to local machine.
#fetch files by appending curl to hyperlink and then using bash script to actually execute curl command and fetch files and redirect/save to the actual filename with extension on local machine.


from __future__ import division
import urllib
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext, Row


def main():
    sc = SparkContext(conf = SparkConf().setAppName("Fetch training data to local"))
    labels = sc.textFile("/Users/priyanka/Desktop/X_train_small.txt")
    
    #To fetch test data comment above labels and uncomment below line.
    #labels = sc.textFile("/Users/priyanka/Desktop/X_test_small.txt")
    
    #O/P: curl https://s3.amazonaws.com/eds-uga-csci8360/data/project2/metadata/filename.asm > filename.asm
    urlAsm = labels.map(lambda x: "curl https://s3.amazonaws.com/eds-uga-csci8360/data/project2/metadata/" + x +".asm>"+ x+".asm").coalesce(1).saveAsTextFile("/Users/priyanka/Desktop/linksasm.txt")
    
    #then use links/part-00000 rename as linksasm.sh, chmod +x linksasm.sh , move file to destination folder and ./linksasm.sh

    #O/P: curl https://s3.amazonaws.com/eds-uga-csci8360/data/project2/metadata/filename.byte > filename.asm
    urlByte = labels.map(lambda x: "curl https://s3.amazonaws.com/eds-uga-csci8360/data/project2/binaries/" + x +".byte>"+ x+".byte").coalesce(1).saveAsTextFile("/Users/priyanka/Desktop/linksbytes.txt")

    #After this step, rename links/part-0000 as Bytelinks.sh, chmod +x Bytelinks.sh, move file to destination folder and ./Bytelinks.sh