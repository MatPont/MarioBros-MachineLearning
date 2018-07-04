# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import sys

"""
fileName = sys.argv[1]

fichier = open(fileName, "r")
content = fichier.read()
fichier.close()
lines = content.split("\n")
result = []
for e in lines:
	result.append(e.split(","))
"""

if len(sys.argv) < 2:
	print("Usage:",sys.argv[0],"fileName...")
	sys.exit(1)

filesResult = []

for arg in sys.argv[1:]:
	#fileName = "episode_values.txt"
	fileName = arg
	fichier = open(fileName, "r")
	content = fichier.read()
	fichier.close()
	lines = content.split("\n")
	result = []
	for e in lines:
		result.append(e.split(","))
	filesResult.append(result)


#--- Functions ---
def getWinLoss():
	print("\nGetWinLoss")
	moyWin = 0
	for r in range(len(filesResult)):
		nbWin = 0
		nbLoss = 0	
		result = filesResult[r]
		for i in range(len(result)):
			if(result[i][0] == "WIN "):
				nbWin += 1
			elif(result[i][0] == "LOSS"):
				nbLoss +=1
		print("",sys.argv[r+1])
		print(" - Win : ",nbWin)
		print(" - Loss: ",nbLoss)	
		moyWin += nbWin
	moyWin /= len(filesResult)
	print(" Mean:")	
	print(" - Win : ",moyWin)
	input()	
	
def getPlot(column, index):
	index = int(index)
	column = int(column)
	myRes = []
	result = filesResult[index]
	print("\nGetPlot")
	for i in range(len(result)):
		if(result[i][0] == "WIN " or result[i][0] == "LOSS"):
			if(column != 0):
				myRes.append(float(result[i][column]))
			else:
				myRes.append(1 if result[i][0] == "WIN " else 0)
	time = [x for x in range(len(myRes))]			
	plt.plot(time, myRes)
	plt.show()
	
def get3MPlot(column):
	column = int(column)
	meanRes = []
	maxRes = []
	minRes = []		
	print("\nGetPlot")
	for i in range(len(filesResult[0])):
		myRes = []		
		for r in range(len(filesResult)):
			result = filesResult[r]	
			if(result[i][0] == "WIN " or result[i][0] == "LOSS"):
				if(column != 0):
					myRes.append(float(result[i][column]))
				else:
					myRes.append(1 if result[i][0] == "WIN " else 0)
		if len(myRes) is not 0:					
			meanRes.append(np.mean(myRes))
			maxRes.append(np.amax(myRes, axis=0))
			minRes.append(np.amin(myRes, axis=0))
	time = [x for x in range(len(meanRes))]			
	plt.plot(time, meanRes, label="mean")
	plt.plot(time, maxRes, label="max")
	plt.plot(time, minRes, label="min")
	plt.legend()
	print(np.mean(meanRes))
	plt.show()
	
def listFiles(preText):
	for arg, i in zip(sys.argv[1:], range(len(sys.argv))):
		print(preText,i,"-",arg)
#-----------------

choice = 1
while(choice != '0'):
	print("""
----------------------------------------------------------------------------------------
0: [WIN/LOSS]     | 1: total_reward   | 2: min_reward | 3: max_reward    | 4: avg_reward
5: num_iterations | 6: mario_position | 7: time_left  | 8: ennemy_killed | 9: [Special]
----------------------------------------------------------------------------------------""")
	print("\n0. Quit\n1. Get Win/Loss")
	print("2. Get plot for nth column\n3. Get mean/max/min plot for nth column")

	choice = input("\nEnter a number: ")
	if(choice == '1'):
		getWinLoss()
	elif(choice == '2'):
		if(len(sys.argv) != 2):
			listFiles("_")
			numFile = input("- Enter a file number: ")
		else:
			numFile = 0
		column = input("- Enter a column index: ")
		getPlot(column, numFile)
	elif(choice == '3'):
		column = input("- Enter a column index: ")	
		get3MPlot(column)
	
		
