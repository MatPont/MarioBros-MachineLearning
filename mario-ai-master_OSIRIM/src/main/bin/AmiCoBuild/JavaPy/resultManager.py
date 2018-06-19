# coding: utf-8

import matplotlib.pyplot as plt

fileName = "episode_values.txt"

lineSize = 9

fichier = open(fileName, "r")
content = fichier.read()
fichier.close()
lines = content.split("\n")
result = []
for e in lines:
	result.append(e.split(","))
	
#print(result)

#--- Functions ---
def getWinLoss():
	print("\nGetWinLoss")
	nbWin = 0
	nbLoss = 0	
	for i in range(len(result)):
		if(result[i][0] == "WIN "):
			nbWin += 1
		elif(result[i][0] == "LOSS"):
			nbLoss +=1
	print(" - Win : ",nbWin)
	print(" - Loss: ",nbLoss)	
	input()	
	
def getPlot(column):
	column = int(column)
	myRes = []
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
#-----------------

choice = 1
while(choice != '0'):
	print("""
----------------------------------------------------------------------------------------
0: [WIN/LOSS]     | 1: total_reward   | 2: min_reward | 3: max_reward    | 4: avg_reward
5: num_iterations | 6: mario_position | 7: time_left  | 8: ennemy_killed | 9: [Special]
----------------------------------------------------------------------------------------""")
	print("\n1. Get Win/Loss\n2. Get plot for nth column\n3.")

	choice = input("\nEnter a number: ")
	if(choice == '1'):
		getWinLoss()
	if(choice == '2'):
		column = input("- Enter a column index: ")
		getPlot(column)
	
		
