import csv
import math
import random as rand

full = []  # raw data for each participant

# this is the parser
def initial():
    with open('Project1_data.csv') as f:
        with open('arsedData.csv', 'w') as out:
            writer = csv.writer(out)
            csvreader = csv.reader(f)
            for row in csvreader:
                use = False
                mara = [False, 0, False, 0, False, 0, False, 0]  # did event, in order 2012..2015 followed by time
                halfmara = [False, 0, False, 0, False, 0, False, 0]
                races = (len(row) - 1) / 5
                gender = "male"
                age = 0
                for i in range(0, races):
                    if ("Marathon Oasis" in row[5 * i + 2]):
                        if ("Demi" in row[5 * i + 3]):  # half or point marathon
                            use = True
                            if ("2012" in row[5 * i + 1]):  # year
                                halfmara[0] = True
                                if '-' in row[5 * i + 3]:
                                    halfmara[1] = -1.0
                                else:
                                    halfmara[1] = convertTime(row[5 * i + 4])
                            elif ("2013" in row[5 * i + 1]):
                                halfmara[2] = True
                                if '-' in row[5 * i + 3]:
                                    halfmara[3] = -1.0
                                else:
                                    halfmara[3] = convertTime(row[5 * i + 4])
                            elif ("2014" in row[5 * i + 1]):
                                halfmara[4] = True
                                if '-' in row[5 * i + 3]:
                                    halfmara[5] = -1.0
                                else:
                                    halfmara[5] = convertTime(row[5 * i + 4])
                            elif ("2015" in row[5 * i + 1]):
                                halfmara[6] = True
                                if '-' in row[5 * i + 3]:
                                    halfmara[7] = -1.0
                                else:
                                    halfmara[7] = convertTime(row[5 * i + 4])
                        elif ("Marathon" in row[5 * i + 3]):

                            use = True
                            if ("2012" in row[5 * i + 1]):
                                mara[0] = True
                                if '-' in row[5 * i + 3]:
                                    mara[1] = -1.0
                                else:
                                    mara[1] = convertTime(row[5 * i + 4])
                            elif ("2013" in row[5 * i + 1]):
                                mara[2] = True
                                if '-' in row[5 * i + 3]:
                                    mara[3] = -1.0
                                else:
                                    mara[3] = convertTime(row[5 * i + 4])
                            elif ("2014" in row[5 * i + 1]):
                                mara[4] = True
                                if '-' in row[5 * i + 3]:
                                    mara[5] = -1.0
                                else:
                                    mara[5] = convertTime(row[5 * i + 4])
                            elif ("2015" in row[5 * i + 1]):
                                mara[6] = True
                                if '-' in row[5 * i + 3]:
                                    mara[7] = -1.0
                                else:
                                    mara[7] = convertTime(row[5 * i + 4])
                        if ("F" in row[5 * i + 5]) or ("f" in row[5 * i + 5]):  # gender
                            gender = "female"
                        if len(row[5 * i + 5].split('-')) == 2:
                            try:  # age
                                age = max(age, int(row[5 * i + 5].split('-')[1]))
                            except ValueError:
                                try:
                                    age = max(int(filter(lambda x: x.isdigit(), row[5 * i + 5])), age)
                                except ValueError:
                                    age = 0
                                    #print "age not changed"

                        else:
                            try:
                                age = max(int(filter(lambda x: x.isdigit(), row[5 * i + 5])), age)
                            except ValueError:
                                age = 0
                                #print "age not changed"

                point = [row[0], races, gender, age] + mara + halfmara
                
                full.append(point)
            writer.writerows(full)

# converts to hours
def convertTime(time):
    l = time.split(':')
    if int(l[0]) == -1:
        return -1.0
    return float(l[0]) + float(l[1]) / 60.0 + float(l[2]) / 3600.0


def calc_thons(entry, endpt):
    numMthons = 0
    numHthons = 0
    avgHthonTime = 0
    for i in range(2, endpt):
        if (entry[i * 2] == True): numMthons += 1

    for i in range(6, endpt+4):
        if (entry[i * 2] == True):
            numHthons += 1  # half mthon
            avgHthonTime += entry[(i*2)+1]

    if (numHthons > 0): avgHthonTime /= float(numHthons)
    return [numMthons, numHthons, avgHthonTime]


def buildDataPoints(trainX, trainY, testX, testY, numFeatures, normz):
    # builds data from full[] into normalized points for logistic reg

    num_pos = 0
    for entry in full:
        numRaces = entry[1]

        if (entry[2] == 'male'):
            sex = 0
        else:
            sex = 1
        age = entry[3]


        #first for test data
        numMthons = 0

        for i in range (2,5):
            if (entry[i*2] == True): numMthons +=1

        hthons =  calc_thons(entry, 6)
        numHthons = hthons[1]
        hthon_time = hthons[2]

        data = [numRaces, age, numHthons, float(entry[8]) , float(entry[10]), float(entry[11]), numMthons, pow(numRaces,2), pow(age,2), pow(numMthons,2), float(entry[6]), float(entry[7]), float(entry[9]), hthon_time, sex]
        testX.append(data[:numFeatures])

        if (numMthons > 1):
            thons = calc_thons(entry, 4)
            numMthons = thons[0]
            numHthons = thons[1]
            hthon_time = thons[2]

            data = [numRaces, age, numHthons, float(entry[6]), float(entry[8]), float(entry[9]), numMthons, pow(numRaces,2), pow(age,2), pow(numMthons,2), float(entry[4]), float(entry[5]), float(entry[7]), hthon_time, sex]
            trainX.append(data[:numFeatures])
            trainY.append(entry[10])



    print "Percent who run again, given they have run in a previous year: " + str(len(trainX)/float(len(testX)))


    if (normz == True):
        basicNormz(trainX)
        basicNormz(testX)


def basicNormz(set):
    for i in range (0, len(set[0])):
        norm_min = 100000
        norm_max = 0
        for datapoint in set:
            norm_min = min(norm_min, datapoint[i])
            norm_max = max(norm_max, datapoint[i])

        for datapoint in set:
            if (norm_max-norm_min != 0):    datapoint[i] = float(datapoint[i] - norm_min)/ float(norm_max - norm_min)

def stdNormz(set):
    for i in range (0, len(set[0])):
        avg = 0
        var = 0
        for datapoint in set:
            avg += datapoint[i]
        avg /= float(len(set))
        for datapoint in set:
            var += pow(datapoint[i]-avg, 2)
        for datapoint in set:
            datapoint[i] = (datapoint[i] - avg) / var



def trainLogReg(trainingSet, Y, weights, err_thresh, max_epoch, acc, lamb, bias):  #maybe dataset param, or just use Xdata

    if (len(trainingSet[0]) != len(weights)):
        print "ERROR: feature len != w len"
        return -1

    err = 100000
    epoch = 0

    for w in weights:
        w = (rand.random() * 2)-1

    while (err > err_thresh and epoch < max_epoch):
        #fwd run
        y_out = []
        for i in range(0, len(trainingSet)):
            y_out.append(logReg(trainingSet[i], weights, bias))

        #grad desc
        err = 0
        wchg = [0]*len(weights)
        for j in range (0, len(y_out)):
            for i in range (0, len(weights)):
                wchg[i] += trainingSet[j][i]*(Y[j]-y_out[j]) + lamb*(weights[i])
            if (y_out[j] != 1 and y_out[j] != 0):
                err += abs(Y[j]*math.log(y_out[j]) + (1-Y[j])*math.log(1-y_out[j]))
            elif (y_out[j] != 0): err += abs(Y[j]*math.log(y_out[j]))
            else: err += abs((1-Y[j])*math.log(1-y_out[j]))
        err /= len(trainingSet)  #ie roughly a % for readability
        #if (epoch % 40 == 0): print "err = " + str(err) + " at epoch " + str(epoch)

        for i in range (0, len(weights)):
            weights[i] += (acc)*wchg[i]
        epoch += 1

    return err


def logReg(x, w, bias):
    ans=0

    for i in range (0, len(x)):
        ans += w[i]*x[i]

    ans += bias

    if (ans < -500):
        ans = 0
        #for i in range(0, len(x)): print str(w[i]) + " , " + str(x[i])
        #print "WARNING: Answer extremely negative."

    else: ans = 1/(1+math.exp(-ans))
    #print ans
    return ans

def logReg_predict(x, w,bias):
    ans=0

    for i in range (0, len(x)):
        ans += w[i]*x[i]

    ans+=bias

    if (ans < -500):
        ans = 0

    else: ans = 1/(1+math.exp(-ans))
    ans = round(ans)
    return ans

def test(valdnSet, Y, w, bias):
    err = 0
    true_pos = 0
    false_neg = 0
    false_pos = 0

    for i in range (0, len(valdnSet)):
        ans = logReg(valdnSet[i], w, bias)
        if (round(ans) != Y[i]): err += 1
        if (round(ans) == 1 and Y[i] == 1): true_pos +=1
        elif (Y[i] == 0): false_pos +=1
        elif (round(ans) == 0 and Y[i] == 1): false_neg +=1

    recall = true_pos / float(true_pos + false_neg)
    precision = true_pos/ float(true_pos + false_pos)
    #print "true+ = " + str(true_pos) + " false- = " + str(false_neg) + " false+ = " + str(false_pos)
    err /= float(len(valdnSet))
    return [err, recall, precision]



def kfold_valdn(X, Y, weights, k, err_thresh, max_epoch, acc, lamb, bias):

    cutoff = int(len(X) / k) #might lose some datapoints due to rounding (only 2 i think)
    valdErr = 0
    trainErr = 0
    recall = 0
    precision = 0
    for i in range (0, k):
        trainLogReg( X[:i*cutoff] + X[(i+1)*cutoff:], Y[:i*cutoff] + Y[(i+1)*cutoff:], weights, err_thresh, max_epoch, acc, lamb, bias)
        trainResult = test(X[:i*cutoff] + X[(i+1)*cutoff:], Y[:i*cutoff] + Y[(i+1)*cutoff:], weights, bias)
        trainErr += trainResult[0]
        valdResult = test(X[i*cutoff:(i+1)*cutoff], Y[i*cutoff:(i+1)*cutoff], weights, bias)
        valdErr += valdResult[0]
        recall += valdResult[1]
        precision += valdResult[2]
        #print "finished kfold vald # " + str(i) + " with trainErr = " + str(trainErr) + " and valdErr= " + str(valdErr)
    valdErr /= float(k)
    trainErr /= float(k)
    recall /= float(k)
    precision /= float(k)
    return [valdErr, trainErr, recall, precision]

def run(numFeatures, k, err_thresh, max_epoch, acc, lamb, normz, bias):
    trainX = []
    trainY = []
    testX = []
    testY = []

    buildDataPoints(trainX, trainY, testX, testY, numFeatures, normz)  # 11 is all

    # init weights
    w = []
    for i in range(0, numFeatures):
        w.append((rand.random() * 2) - 1)

    #print "Num Features = " + str(numFeatures)
    err = kfold_valdn(trainX, trainY, w, k, err_thresh, max_epoch, acc, lamb, bias)
    print "validation err: " + str(err[0])
    print "training err: " + str(err[1])

    trainLogReg(trainX, trainY, w, err_thresh, max_epoch, acc, lamb, bias)
    print "final weights: " + str(w)

    true_count = 0

    for x in testX:
        ans = logReg_predict(x, w, bias)
        if (ans == 0): ans = 0
        elif (ans == 1):
            ans = 1
            true_count +=1
        else: print "ERR: Ans not 0 or 1"
        testY.append([ans])

    print "# true = " + str(true_count) + " out of " + str(len(testX))
    print "% true = " + str((true_count)/float(len(testX)))
    # write to prediction file
    


    with open("LogReg_predictions.csv",'wb') as l:
        writer=csv.writer(l,delimiter=',')
        writer.writerows(testY)
def main():
    '''
    initial()

    print "\nNothing Control"
    run(0, 8, .001, 500, 0, 0, True, 0)

    print "\n+ Bias Control"
    run(0,8, .001, 500, .005, 0, True, .5)

    print "\n- Bias Control"
    run(0,8, .001, 500, .005, 0, True, -.5)

    print "\nNo Acc Control"
    run(6, 8, .001, 500, 0, 0, True, .5)

    print "\nNo Normz"
    run(6, 8, .001, 500, .005, 0, False, .5)

    print "\nNo Norm, Regn"
    run(6, 8, .001, 500, .005, -.000075, False,.5)

    print "\nNorm, no Regn"
    run(6, 8, .001, 500, .005, 0, True, .5)

    print "\nNorm and Regn"
    run(6, 8, .001, 500, .005, -.000075, True, .5)
    '''
    initial()
    run(6, 8, .001, 500, .001, -.000075, True, .5)
    #numFeatures, #-fold validation, err_threshold, epochs, learning-rate, lambda regularization, normalize?, bias

main()




