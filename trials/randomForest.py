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


    
#cleans /r,/n and number id in beginning of each line automatically
def clean(x):
    l= []
    for item in x:
        if len(item) <= 2:
            l.append(item)
    return l       

    
    
def main():
    sc = SparkContext(conf = SparkConf().setAppName("Random Forest"))
    bytePath = "/Users/priyanka/Desktop/project2files/train"
    byteTestPath = "/Users/priyanka/Desktop/project2files/test"
    namePath = "/Users/priyanka/Desktop/X_train_small.txt"
    nameTestPath="/Users/priyanka/Desktop/X_test_small.txt"
    classPath = "/Users/priyanka/Desktop/y_train_small.txt"
    classTestPath = "/Users/priyanka/Desktop/y_test_small.txt"
    docData= sc.wholeTextFiles(bytePath, 25).map(lambda (x,y):(x.encode("utf-8"),y.encode("utf-8")))
    print ("docData frankie")
    docData.take(1)
    #clean docData here - remove 1st word from line and remove  
    cleanDocData = docData.map(lambda (x,y): (x,clean(y.split())))
    nameData = sc.textFile(namePath,25).map(lambda x: "file:"+bytePath+"/"+x+".bytes").zipWithIndex().map(lambda (x,y):(y,x))
    #nameData.take(5)   
    labelData = sc.textFile(classPath,25).zipWithIndex().map(lambda (x,y):(y,str(int(x)-1)))
    joinNameLabel = nameData.join(labelData).map(lambda (x,y):y)
    #joinNameLabel.take(5) 
    joinCleanDocLabel = joinNameLabel.join(cleanDocData).map(lambda (x,y):y)
    print ("starting hashingTF rosie")
    #joinCleanDocLabel.persist()
    #tfFormat = joinDocLabel.map(lambda (x,y): (x,y.split()))
    hashingTF = HashingTF()    
    hashData = joinCleanDocLabel.map(lambda (label, text): LabeledPoint(label, hashingTF.transform(text)))
    hashData.persist()
    print "hashing TF done"
    #    model = NaiveBayes.train(hashData)
    
    print ("generating model fliss")
    model1 = RandomForest.trainClassifier(hashData, numClasses=10, categoricalFeaturesInfo={},
                                         numTrees=50, featureSubsetStrategy="auto",
                                         impurity='gini', maxDepth=4, maxBins=32)
    
    #==============================================================================
    # Testing starts here
    #==============================================================================
    docTestData = sc.wholeTextFiles(byteTestPath, 25).map(lambda (x,y):(x.encode("utf-8"),y.encode("utf-8")))
    #docTestData.take(1)
    
    cleanDocTestData = docTestData.map(lambda (x,y): (x,clean(y.split())))
    nameTestData=sc.textFile(nameTestPath,25).map(lambda x: "file:"+byteTestPath+"/"+x+".bytes").zipWithIndex().map(lambda (x,y):(y,x))
    labelTestData = sc.textFile(classTestPath,25).zipWithIndex().map(lambda (x,y):(y,str(int(x)-1)))
    #joinTestNameLabel=nameTestData.join(labelTestData).map(lambda (x,y):y).cache()
    
    joinTestDocLabel=joinTestNameLabel.join(cleanDocTestData).map(lambda (x,y):y)
    #joinTestDocLabel.persist()
    #tfTestFormat = joinTestDocLabel.map(lambda (x,y): (x,y.split()))
    print ("hashing test kenny")
    hashTestData = joinTestDocLabel.map(lambda (label, text): LabeledPoint(label, hashingTF.transform(text)))
    sc.broadcast(hashTestData)
    
    #    #NB prediction and labels and accuracy
    #    prediction_and_labels = hashTestData.map(lambda point: (model.predict(point.features), point.label))
    #    correct = prediction_and_labels.filter(lambda (predicted, actual): predicted == actual)
    #    accuracy = correct.count() / float(hashTestData.count())
    #    print "Classifier correctly predicted category " + str(accuracy * 100) + " percent of the time"
    
    
    #Random forest prediction and labels and accuracy
    print "prediction part lyndz"
    prediction1 = model1.predict(hashTestData.map(lambda x: x.features))
    labelsAndPredictions1 = hashTestData.map(lambda lp: lp.label).zip(prediction1)
    testErr1 = labelsAndPredictions1.filter(lambda (v, p): v != p).count() / float(hashTestData.count())
    print('Test Error = ' + str(testErr1))
    print('Learned classification forest model:')
    print(model.toDebugString())
    
    # Save and load random forest model
    model1.save(sc, "/Users/priyanka/Desktop/project2files/myRandomForestClassificationModel")
    sameModel1 = RandomForestModel.load(sc, "/Users/priyanka/Desktop/project2files/myRandomForestClassificationModel")
    
    
    
    #save and load NB model
    #    
    #    output_dir = '/Users/priyanka/Desktop/project2files/myNaiveBayesModel'
    #    shutil.rmtree(output_dir, ignore_errors=True)
    #    model.save(sc, output_dir)
    #    sameModel = NaiveBayesModel.load(sc, output_dir)
    #    predictionAndLabel = test.map(lambda p: (sameModel.predict(p.features), p.label))
    #    accuracy = 1.0 * predictionAndLabel.filter(lambda (x, v): x == v).count() / test.count()
    #    print('sameModel accuracy {}'.format(accuracy))
        
    
    
if __name__ == "__main__":
    main()
