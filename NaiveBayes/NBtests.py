import csv
import math
import random
import pandas as pd

# randomly splits the dataset in trainSet and testSet with a ratio of splitRatio
def splitDataset(dataset, splitRatio):
    trainSize = int(len(dataset) * splitRatio)
    trainSet = []
    testSet = list(dataset)
    while len(trainSet) < trainSize:
        index = random.randrange(len(testSet))
        trainSet.append(testSet.pop(index))
    return [trainSet, testSet]

def mean(numbers):
    return sum(numbers)/float(len(numbers))

def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
    return math.sqrt(variance)

"""Input: filename
   Output:  
   dataset = contains well formed dataset, 
   X feature matrix,
   Y label vector, 
   columns names parameters of columns
"""
def getDataReady(filename):
    with open(filename, 'rb') as csvfile:
        reader = csv.reader(csvfile)
        dataset = []
        for row in reader:
            for i in range(len(row)):
                if (row[i] == 'male') or (row[i] == 'True') :
                    row[i] = 1
                elif (row[i] == 'female') or (row[i] == 'False'):
                    row[i] = 0
            del row[0] # this deletes the participant ID number
            dataset.append(row) 
        X = []
        y = []
        columns = dataset[0] # deletes the names of the columns
        del dataset[0]
        for row in dataset:
            for i in range(len(row)):
                row[i] = float(row[i])
            X.append(row[0:-1])
            y.append(row[-1])
        return dataset,X,y,columns
# for a feature xi calculate P(xi|y = participate) and P(xi|y= not participate)
# input dataset, index of xi feature and index of y class 
# returns the C1: p(y=1)*p(xi=1|y=1) and C1: p(y=0)*p(xi=1|y=0)
def getProbBinary(dataset,xi,y):
    y0 = 0.0
    y1 = 0.0
    x1y1 = 0.0
    x1y0 = 0.0
    for row in dataset:
        if row[y] == 0.0:
            y0 = y0+1
        if row[y] == 1.0:
            y1 = y1+1
        if row[y] == 1.0 and row[xi] == 1.0:
            x1y1 = x1y1 + 1
        if row[y] == 0.0 and row[xi] == 1.0:
            x1y0 = x1y0 + 1
    px1y1 = (x1y1)/(y1)
    px1y0 = (x1y0)/(y0)
    return px1y0,px1y1

def getProbabilitiesContinuous(summaries, inputRow):
    probabilities = {}
    for classValue, classFeatures in summaries.iteritems():
        probabilities[classValue] = 1
        for i in range(len(classFeatures)):
            mean, stdev = classFeatures[i]
            x = inputRow[i]
            probabilities[classValue] *= normalPDF(x, mean, stdev)
    return probabilities

# get the probabilty given a number x
def normalPDF(x,mean,stdev):
    exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
    return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

# for feature xi 
# input= dataset, feature index xi, label index y
# return= Y=0 : [mean,stdev] Y:1 [mean,stdev]
def getMeanStd(dataset,xi,y):
    x1s = []
    x0s = []
    for row in dataset:
        if row[y] == 1.0:
            x1s.append(row[xi])
        if row[y] == 0.0:
            x0s.append(row[xi])
    return [mean(x0s),stdev(x0s)],[mean(x1s),stdev(x1s)]

def predictBinary(dictBinary,inputRow):
    probabilities = {}
    for classValue, classFeatures in dictBinary.iteritems():
        probabilities[classValue] = 1
        for i in range(len(classFeatures)):
            if inputRow[i] == 1:
                probabilities[classValue] *= classFeatures[i]
            if inputRow[i] == 0:
                probabilities[classValue] *= (1-classFeatures[i])
    return probabilities

def predict(dBinary,dCont):
    if dBinary[0]*dCont[0] > dCont[1]*dBinary[1]:
        return 0
    if dBinary[0]*dCont[0] < dCont[1]*dBinary[1]:
        return 1

def getPredictions(testSet,continuousSummary,binaryProbabilities):
    predictions = []
    for row in testSet:
        dBinary = predictBinary(binaryProbabilities,row[2:len(testSet[0])])
        dContinuous = getProbabilitiesContinuous(continuousSummary,row[0:2])
        aPrediction = predict(dBinary,dContinuous)
        predictions.append(aPrediction)
    return predictions

def predictContinuous(dContinuous):
    if dContinuous[0] > dContinuous[1]:
        return 0
    if dContinuous[0] > dContinuous[1]:
        return 1
    
def getPredicitonOnlyCont(testSet,continuousSummary):
    predicitons = []
    for row in testSet:
        dContinuous = getProbabilitiesContinuous(continuousSummary,row[0:2])
        aPrediction = predictContinuous(dContinuous)
        predicitons.append(aPrediction)
    return predictions

def accuracy(testSet,predictions):
    count = 0
    for i in range(len(testSet)):
        if testSet[i][-1] == predictions[i]:
            count += 1
    return (count/float(len(testSet))) * 100.0

def main():
    filename = './parsedDataNoTime.csv'
    dataset,X,y,c = getDataReady(filename)
    accuracies = []
    # race1-4 2012-2015 full 
    # race5-6 2012-2015 1/2
    # test number races age race 3 race 4
    for i in range(0,100):
        trainSet, testSet = splitDataset(dataset,0.6)
        summaryContinuous = {0:[],1:[]}
        binaryProb = {0:[], 1:[]}
        for i in range(0,2):
            [m0,s0],[m1,s1] = getMeanStd(dataset,i,-1)
            summaryContinuous[0].append([m0,s0])
            summaryContinuous[1].append([m1,s1])
        indexes = [2,-2]
        for i in range(2,len(dataset[0])-1):
            class0,class1 = getProbBinary(dataset,i,-1) 
            binaryProb[0].append(class0)
            binaryProb[1].append(class1)
        predictions = getPredictions(testSet,summaryContinuous,binaryProb)
        accuracies.append(accuracy(testSet,predictions))
    print mean(accuracies)
    print stdev(accuracies)

main()