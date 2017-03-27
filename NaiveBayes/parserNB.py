import csv
import numpy as np 
import math 

full = [] 

# converts to hours
def convertTime(time): 
	l = time.split(':')
	if int(l[0]) == -1: 
		return -1.0
	return float(l[0]) + float(l[1])/60.0 + float(l[2])/3600.0


# this function was written by Ed, I'm gonna edit it so that it gives me 
# the data in a form that will be easier to use with NaiveBayes
def initial():  
	with open('./Project1_data.csv') as f:
		with open('./parsedDataNaiveBayes.csv', 'w') as out:
			writer = csv.writer(out, delimiter=',', quoting=csv.QUOTE_MINIMAL)
			writer.writerow(['participantId','numberOfRaces','sex','age','race1','time1','race2','time2','race3','time3','race4','time4','race5','time5','race6','time6','race7','time7','race8','time8'])
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
									print "age not changed"

						else:
							try:
								age = max(int(filter(lambda x: x.isdigit(), row[5*i+5])), age)
							except ValueError:
								print "age not changed"

				point = [row[0], races, gender, age] + mara + halfmara
				if use == True: 
					full.append(point)
			writer.writerows(full)
initial()
# First we have to hand the data Ed has parsed. 
# parsedDataFinal.csv -> 
import pandas as pd
df = pd.read_csv('./parsedDataNaiveBayes.csv')
#df.head()
# We the order of races is in the following way
# race1-4 2012-2015 full 
# race5-6 2012-2015 1/2
# test number races age race 3 race 4
# It is placed like this to make it easier with array indicies
df2 = df[['participantId','numberOfRaces','age','sex','race5','race6','race7','race8','race1','race2','race3','race4']]
#df2.head()
df2.to_csv('./parsedDataNoTime.csv',index=False)