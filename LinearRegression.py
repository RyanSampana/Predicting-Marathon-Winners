import csv
import numpy as np 
import math 
full = [] # raw data for each participant 
full =[]
Xdata = [] # final set of X data points 
Ydata =[]	# final set of Y data points
thresh = 0.000001 # threshold at which to stop for gradient decent 
regu = False # boolean for whether to use regularization 
useGrad = False	# boolean for whether to use gradient descent 
lam = 0.01 # regularization hyperparameter
a=0.0000001 # initial learning rate for gradient descent 
Xr = 0
#fucntion to preform linear regression, input is and numpy X and Y matrix 
def regress(X,Y):
	Wcur = np.matrix( np.ones(np.shape(X)[1]) ).T
	Wpre = np.matrix( np.ones(np.shape(X)[1]) ).T
	[Wcur,err]= grad(X,Y,Wcur,0.0000001,lam)
	if(useGrad):
		k=0
		while( np.linalg.norm( Wcur- Wpre) > thresh ):
			Wpre = Wcur
			k+=1
			[Wcur,err] = grad(X,Y,Wcur,1/((1/a) + k),lam)
		return Wcur
	else:
	 	w = (X.T*X).I*X.T*Y
	 	return w

# fucntion to preform gradient descent 
def grad(X,Y,w,a,lam):
	if (regu):
		derr = 2.0*((X.T)*(X)*w - (X.T)*Y) + 2.0*lam*w
	else:  
		derr = 2.0*((X.T)*(X)*w - (X.T)*Y) 
	return [(w - a*derr), derr]

#this is the parser, converts the database to raw data and puts in the variable "full"
# once called "full" will be of the form 
# [participant #, # of races run, gender, age, boolean for ran 2012 Marathon, time of 2012 Marathon, boolean for ran 2013 Marathon, time of 2013 Marathon,..., boolean for ran 2012 Half-Marathon, time of 2012 Half-Marathon,...] 
def initial():  
	with open('Project1_data.csv') as f:
		csvreader = csv.reader(f)
		for row in csvreader:
			use = False
			mara = [False, 0, False, 0 , False, 0, False, 0]						# did event, in order 2012..2015 followed by time 
			halfmara = [False, 0, False, 0 , False, 0, False, 0]
			races = (len(row) - 1) /5
			gender = "male"
			age = 0
			for i in range(0, races): 
				if ("Marathon Oasis" in row[5*i + 2]): 
					if ("Demi" in row[5*i + 3]): 					#half or point marathon
						use = True
						if( "2012" in row[5*i + 1]):				#year 
							halfmara[0] = True
							if '-' in row[5*i + 3]:
								halfmara[1] = -1.0 
							else: 
								halfmara[1] = convertTime(row[5*i + 4])
						elif( "2013" in row[5*i + 1]):
							halfmara[2] = True
							if '-' in row[5*i + 3]:
								halfmara[3] = -1.0
							else: 
								halfmara[3] = convertTime(row[5*i + 4])
						elif( "2014" in row[5*i + 1]):
							halfmara[4] = True
							if '-' in row[5*i + 3]:
								halfmara[5] = -1.0 
							else: 
								halfmara[5] = convertTime(row[5*i + 4])
						elif( "2015" in row[5*i + 1]):
							halfmara[6] = True
							if '-' in row[5*i + 3]:
								halfmara[7] = -1.0 
							else: 
								halfmara[7] = convertTime(row[5*i + 4])
					elif("Marathon" in row[5*i + 3]):
						
						use = True
						if( "2012" in row[5*i + 1]):
							mara[0] = True
							if '-' in row[5*i + 3]:
								mara[1] = -1.0
							else: 
								mara[1] = convertTime(row[5*i + 4])
						elif( "2013" in row[5*i + 1]):
							mara[2] = True
							if '-' in row[5*i + 3]:
								mara[3] = -1.0 
							else: 
								mara[3] = convertTime(row[5*i + 4])
						elif( "2014" in row[5*i + 1]):
							mara[4] = True
							if '-' in row[5*i + 3]:
								mara[5] = -1.0
							else: 
								mara[5] = convertTime(row[5*i + 4])
						elif( "2015" in row[5*i + 1]):
							mara[6] = True
							if '-' in row[5*i + 3]:
								mara[7] = -1.0
							else: 
								mara[7] = convertTime(row[5*i + 4])
					if ("F" in row[5*i + 5]) or ("f" in row[5*i + 5]):		#gender
						gender = "female"					
					if len(row[5*i+5].split('-')) == 2:		
						try: 				#age
							age = max(age, int(row[5*i+5].split('-')[1]))
						except ValueError:
							try:
								age = max(int(filter(lambda x: x.isdigit(), row[5*i+5])), age)
							except ValueError:
								r = 0
					else:
						try:
							age = max(int(filter(lambda x: x.isdigit(), row[5*i+5])), age)
						except ValueError:
							r=0
							

			point = [row[0], races, gender, age] + mara + halfmara
			if use == True: 
				full.append(point)


#converts the parsed data into the desired form to perfom linear regression
# returns two lists, Xdata and Ydata 
def convert():		 
	for point in full:
		count = 0
		for i in range(0,4):
			if (point[4+2*i] == True):
				count+=1 
		if count <2: 
			continue  
		for i in range(1,4):
			if point[4+(i*2)] == False:
				continue
			Fe = 0
			Ma = 0 
			if  "f" in point[2]:
				Fe = 1
			else:
				Ma = 1
			tempX = [ 0,0,0,0,0,0, point[1] , Ma, Fe, point[3]]
			tempY = [0]
			count= 0 
			for j in range(0,i):
				if (point[4+2*i] == True):
					count+=1
			if count<1:
				continue
			tempY = point[4+(i*2)+1]
			check = False 
			for j in range(0,i):
				if point[4+2*j] == True:
					check = True 
					tempX[2-(i-1)+j ] = point[4+2*j+1]
			for j in range(0,2):
				if tempX[j+1] == 0:
					tempX[j+1] = tempX[j]
					tempX[j]=0
			if tempX[1] == 0:
				tempX[1] = tempX[0]
				tempX[0] = 0			
			tempX[3] = tempX[2]*tempX[2]
			if check == False: 
				continue 
			
			if (tempX[0] == -1) or  (tempX[1] == -1) or (tempX[2] == -1):
				continue
			sum1 = 1  
			if(tempX[0]) != 0:
				sum1 +=2
			sum2 = 1
			if(point[13]) != 0:
				sum2 +=1
			if(point[15]) != 0:
				sum2 +=1
			if(point[17]) != 0:
				sum2 +=1
			aveRun = (tempX[0]+ tempX[1]+ tempX[2])/sum1
			#define the linear funtion to optimize
			tempX = [ aveRun, math.log(tempX[2]),tempX[2]*tempX[2],tempX[2]*tempX[2]*tempX[2] , Ma,Fe, point[3]]
			Xdata.append(tempX)
			Ydata.append(tempY)

			
	return [Xdata,Ydata] 

# converts string of time to hours
def convertTime(time): 
	l = time.split(':')
	if int(l[0]) == -1: 
		return -1.0
	return float(l[0]) + float(l[1])/60.0 + float(l[2])/3600.0

#returns the avergae distance from correct time and the average error
#input is numpy matricies X,Y,w 
def validate(X,Y,w):
	av =0 
	avSqu=0
	l = 0
	for i in range(0, np.shape(X)[0]):
		if Y[i][0] == -1: 
			l+=1
			
			continue
		#print X[i]*w - Y[i]
		av+=  abs(X[i]*w - Y[i])
		avSqu += abs(X[i]*w - Y[i])*abs(X[i]*w - Y[i])
	print " the average distance from correct time is for fold is: "  + str(av/(len(Y)-l))
	print " the average squared error for fold is: " + str(avSqu/(len(Y)-l))
	return [av/(len(Y)-l),avSqu/(len(Y)-l)]
#cross validation function, will return estimate of true prediction error using k-fold cross validation
#takes as input lists X and Y, and k being the deisred number of cuts
def crossValidate(X,Y,k):
	subLen = int(math.floor(len(Y)/k))
	avrSqu=0
	avrSqu2=0
	for i in range(0,k):
		Xleft= X[:i*subLen]
		Xvalid= X[i*subLen+1:(i+1)*subLen]
		if (i == k-1):
			Xright = X[0:0]
		else:
			Xright = X[(i+1)*subLen +1:]
		Xtrain = Xleft+Xright
		Yleft= Y[:i*subLen]
		Yvalid= Y[i*subLen+1:(i+1)*subLen]
		if (i == k-1):
			Yright = Y[0:0]
		else:
			Yright = Y[(i+1)*subLen +1:]
		Ytrain = Yleft + Yright
		w = regress(np.matrix(Xtrain) ,np.matrix(Ytrain).T)
		print "For fold " +str(i+1)
		print "Training errors: "
		[v,t] =  validate(np.matrix(Xtrain),np.matrix(Ytrain).T, w)

		avrSqu += t 
		print " "
		print "Validation errors: "
		[v2,t2] =  validate(np.matrix(Xvalid),np.matrix(Yvalid).T, w)	
		print "----"
		avrSqu2 += t2 

	print " "
	print"Over all set: "
	print " the average error is on the training set is  "  + str(avrSqu/k)
	print " the average error is on validation set  is "  + str(avrSqu2/k)
#returns guessed times for all participants 
def returnValue(w):		
	out = [] 
	for point in full:
		print point[0]
		Fe = 0
		Ma = 0 
		if  "f" in point[2]:
			Fe = 1
		else:
			Ma = 1

		races = [0,0,0]
		for i in range(0,4):
			if point[11-2*i] == -1:
				continue
			for j in range(0,3):
				if races[2-j]==0:
					races[2-j] = point[11-2*i]
					break
		for i in range(0,4):
			if point[11-2*i] == -1:
				races[2] = 0 
			elif point[11-2*i] == 0:
				continue 
			else:
				 break


		sum1 = 1  
		if(races[0]) != 0:
			sum1 +=2
		aveRun = (races[0]+ races[1]+ races[2])/sum1
		
		if races[2] == 0: 
			log = 10000000
		else:
			log = math.log(races[2])
		Xr = np.matrix([ aveRun, log,races[2]*races[2],races[2]*races[2]*races[2] , Ma,Fe, point[3]])

		if ((Xr*w).tolist()[0][0]>0):
			i = 0
			i += (Xr*w).tolist()[0][0]
			hours = int(i)
			minutes = int((i - float(hours))*60.0)
			seconds = int((((i - float(hours))*60.0)  - float(minutes))*60.0)
			estimate= str(hours) + ':' + str(minutes) + ':' + str(seconds)
			out.append([point[0],estimate])
			print  [point[0],Xr*w]
		else:
			if Ma == 1 :
				out.append([str(point[0]) ,'4:19:27'])
			else:
				out.append([str(point[0]) , '4:44:19'])
			
	with open("outdata.csv",'w') as l:
		writer=csv.writer(l,delimiter=',')
		writer.writerows(out)
	return out

# order of operations
def main(): 
	initial()
	[X,Y]=convert()
	crossValidate(X,Y,5)
	wfinal = regress(np.matrix(X),np.matrix(Y).T)
	output = returnValue(wfinal)
main()		    



