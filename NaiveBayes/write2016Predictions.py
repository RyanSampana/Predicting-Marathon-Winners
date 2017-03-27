import NBfunctions as NB
import csv
def main():
    filename = './parsedDataNoTime.csv'
    dataset,X,y,c = NB.getDataReady(filename)

    # race1-4 2012-2015 full 
    # race5-6 2012-2015 1/2
    # test number races age race 3 race 4
    summaryContinuous = {0:[],1:[]}
    binaryProb = {0:[], 1:[]}

    # look at continuous features
    for i in range(1,3):
            [m0,s0],[m1,s1] = NB.getMeanStd(dataset,i,-1)
            summaryContinuous[0].append([m0,s0])
            summaryContinuous[1].append([m1,s1])

    # look at binary features
    for i in range(3,len(dataset[0])-1):
            class0,class1 = NB.getProbBinary(dataset,i,-1) 
            binaryProb[0].append(class0)
            binaryProb[1].append(class1)

    predictions,idNumber = NB.getPredictions(dataset,summaryContinuous,binaryProb,1,3,(len(dataset[0])-1))
    
    # we turn the idNumber back into an int
    for i in range(len(idNumber)):
        idNumber[i] = int(idNumber[i])
    
    # time to write down out predictions
    with open('./NBpredictions.csv', 'w') as out:
            writer = csv.writer(out, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            outty = [idNumber,predictions]
            writer.writerows(zip(*outty))

        

main()
