# -*- coding: utf-8 -*-

import numpy as np
from PyJavaInit import amiCoSimulator

#----------Parameters----------

# rewardID:
#  0: Reward from Mario's game
#  1: Reward when Mario goes to the right and the top
#  2: Both
rewardID = 1

# stateRepresentationID:			 
#  0: Grid view
#  1: 4 parameters view (Mario's speed on x and y axis, distance to the block in front and bottom)
#  2: Both (TODO - not implemented yet)			 
stateRepresentationID = 1

# saveFiles:
modelCheckpoint = "./modelWeights.ckpt"
hyperparametersFile = "./modelParameters.dqn"
memoryFile = "./modelMemory.dqn"

#-----------------------------

class MarioNEATAgent():
			 
	"""for this __init__ we need to only have the "self" parameter (cf. AmiCoJavaPy)"""
	def __init__(self):
		print("Initialization of Mario NEAT Agent")
		"""print("__networkID = ", networkID)
		print("__rewardID = ", rewardID)
		print("__stateRepresentationID = ", stateRepresentationID)  """		 
		if stateRepresentationID == 0:
			state_size = [19,19,1]
		elif stateRepresentationID == 1:
			state_size = [4]
		"""[LEFT,RIGHT,DOWN,JUMP,SPEED,UP] for Mario AI Benchmark""" 
		actions = [
			#THE FIRST ACTION MUST BE NOOP
			[0,0,0,0,0,0], #0 - no button
			#AFTER THAT, THE ORDER DOES NOT MATTER
			[1,0,0,0,0,0], #1 - left
			[0,1,0,0,0,0], #2 - right
			[0,0,1,0,0,0], #3 - down
			[0,0,0,1,0,0], #4 - jump
			[0,0,0,0,1,0], #5 - run (speed)
			[0,0,0,0,0,1], #6 - up
			[1,0,0,0,1,0], #7 - left run
			[1,0,0,1,0,0], #8 - left jump
			[0,1,0,0,1,0], #9 - right run
			[0,1,0,1,0,0], #10 - right jump
			[1,0,0,1,1,0], #11 - left run jump
			[0,1,0,1,1,0], #12 - right run jump
			[0,0,1,1,0,0], #13 - down jump
		]
		report_frequency=100
		self.agentName = "MarioNEATAgent"
		self.initAgent(state_size, actions, report_frequency)
		print("Initialized - Agent")
		
	def initAgent(self,
					state_size,
					actions,
					report_frequency):
		self.init_new_episode()

		self.state_size = state_size # size of environment state space.
		self.actions = actions # a list of list of NES buttons, each of which is binary and of length 6
		self.chosenAction = None
		self.num_actions = len(actions) # actions available at any state.
		self.cur_state = None
		self.old_state = None
		self.firstFrame = True
		
		#----------Other parameters----------
		
		# Frequencies
		self.compute_action_frequency = 1 # Step before computing another action (4 is often used but isn't giving good results here)

		# and some exploration parameters		
		self.epsilon = 1.0				
		self.epsilon_min = 0.05
		self.epsilon_decay = 0.9999
		
		#------------------------------------	
			
		self.minibatches_run = 0			
	
		# These will hold episode reward information		
		self.episode_count = 0
		self.total_iterations = 0
		self.total_random_action = 0			
		
		self.episode_iterations = 0		
		self.total_episode_reward = 0.
		self.episode_min_reward = None
		self.episode_max_reward = None
		self.episode_time_left = 0.
		self.episode_ennemy_killed = 0		
	
		# How often we get an update printed to screen about how things are looking and what param values are
		self.report_frequency = report_frequency
	
		print("Initialized - NEAT")	
				
	# ----- Choose Action -----
	def choose_action(self, state):
		
		def _choose_to_explore():
			# returns true with a probability equal to that of epsilon
			return np.random.rand() < self.epsilon
		
		def _choose_random_action():
			r = np.random.randint(low=0, high=self.num_actions) # high is 1 above what can be picked
			return self.actions[r]
		
		if _choose_to_explore():
			self.total_random_action += 1
			return _choose_random_action()
		else:
			# TODO - Choose action with NEAT network
			a = 0
											
			return self.actions[a]
			
	def isSkippedFrame(self):
		return self.episode_iterations % self.compute_action_frequency != 0
		#return np.random.rand() < 1. / self.compute_action_frequency
	#------------------------------------------------					

	#------------------------------------------------
	#---------------- Save & Restore ----------------
	#------------------------------------------------		
	def save_episode_values(self, marioState):
		fileName = "episode_values.txt"
		fichier = open(fileName, "a")
		myString = "WIN , " if marioState == 1 else "LOSS, "
		myString += "{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}\n".format(
											self.total_episode_reward,
											self.episode_min_reward,
											self.episode_max_reward,
											self.total_episode_reward / self.episode_iterations,
											self.episode_iterations,
											self.marioFloats[1],
											self.episode_time_left,
											self.episode_ennemy_killed)
		fichier.write(myString)
		fichier.close()
	#------------------------------------------------
						
	def init_new_episode(self):
		self.levelScene = None
		self.mayMarioJump = None
		self.isMarioOnGround = None
		self.marioFloats = None
		self.enemiesFloats = None
		self.marioState = None
		self.levelFinished = False
		self.reward = 0
		self.cumulativeReward = 0
		self.oldIntermediateReward = 0
		self.oldMarioFloats = None
		self.old_state = None
		self.firstFrame = True
		self.report_loss = 0
	
	def start_new_episode(self):
		self.episode_count += 1
		self.episode_iterations = 0
		self.total_episode_reward = 0.
		self.episode_min_reward = None
		self.episode_max_reward = None
		self.episode_time_left = 0.
		self.episode_ennemy_killed = 0		
		self.init_new_episode()							
						
	def print_report(self):
		print("(",self.total_random_action,"/",self.total_iterations,") _ epsilon =",self.epsilon)
		print("reward =",self.reward,"_ total_reward =",self.total_episode_reward)
		
	def printObservation(self, observation):
		print("================================================")
		for i in observation:
			print(i)		
	
	#------------------------------------------------  
	#--------- Mario AI Benchmark functions ---------
	#------------------------------------------------
	"""
	setObservationDetails is calling once at the beginning by the Mario's benchmark then:
	Calling order: giveIntermediateReward, integrateObservation, getAction
	"""
	def giveIntermediateReward(self, reward):
		#print("____giveIntermediateReward")		   
		self.reward = 0
		if(rewardID == 0 or rewardID == 2):
			self.reward += reward - self.oldIntermediateReward
			self.oldIntermediateReward = reward  

	def giveDistanceReward(self):
		coef = 5 
		if((rewardID == 1 or rewardID == 2) and self.oldMarioFloats is not None):
			vx = self.marioFloats[0] - self.oldMarioFloats[0]
			vy = self.marioFloats[1] - self.oldMarioFloats[1]
			tempReward = ( vx - (0.1 * vy) ) * coef
			if(tempReward > 0):
				tempReward += 10
			else:
				tempReward -= 10
			self.reward += tempReward
		#print("reward=",self.reward)
		
	def integrateObservation(self, squashedObservation, squashedEnemies, marioPos, enemiesPos, marioState):
		#print("____integrateObservation")
		if self.firstFrame:
			return
			
		row = self.receptiveFieldHeight
		col = self.receptiveFieldWidth
			
		# Merges both observations
		sceneObservation = []
		test = []
		for i in range(len(squashedObservation)):
			if(squashedEnemies[i] != 0):
				test.append(squashedEnemies[i])
			else:
				test.append(squashedObservation[i])
			if(((i+1) % col) == 0):
				sceneObservation.append(test)
				test = []
		#self.printObservation(sceneObservation)	
		#input()
		sceneObservation = np.array(sceneObservation)
		sceneObservation = np.expand_dims(sceneObservation, axis=2)
		sceneObservation.tolist()
		
		# Update mario's parameters
		self.oldMarioFloats = self.marioFloats
		self.marioFloats = marioPos
		self.enemiesFloats = enemiesPos
		self.mayMarioJump = marioState[3]
		self.isMarioOnGround = marioState[2]
		self.marioState = marioState[1]
		self.levelFinished = (marioState[0] != 2)
		self.episode_time_left = marioState[10]
		self.episode_ennemy_killed = marioState[6]		
		self.levelScene = sceneObservation		
		
		self.episode_iterations += 1		
		
		# Give distance reward (or not, depends on rewardID)
		self.giveDistanceReward()
				
		# Manage skipped frames and cumulative rewards over frames
		self.cumulativeReward += self.reward		
		if self.isSkippedFrame():
			return
		else:
			self.reward = self.cumulativeReward		
			self.cumulativeReward = 0
		
		# Define cur_state according to stateRepresentationID
		if stateRepresentationID == 0 or stateRepresentationID == 2:
			self.cur_state = self.levelScene		
		elif stateRepresentationID == 1:
			self.cur_state = [self.computeXa(), self.computeYa(),
								self.computeForwardBlock(), self.computeBottomBlock()]
		# TODO - Add 4 parameters to cur_state for stateRepresentationID == 2
		"""if stateRepresentationID == 2:	
			myTemp = [self.computeXa(), self.computeYa(),
									self.computeForwardBlock(), self.computeBottomBlock()]
			temp = np.append(self.cur_state, [myTemp])"""

		self.old_state = self.cur_state
		
		# Increment counters for decayed variables
		self.total_iterations += 1	
		
		# Decay agent exploration 
		self.epsilon = max(self.epsilon_min, self.epsilon*self.epsilon_decay)
		
		# The rewards get updated for the episode
		self.total_episode_reward += self.reward											
		if self.episode_min_reward is not None:
			self.episode_min_reward = min(self.reward, self.episode_min_reward)
		else:
			self.episode_min_reward = self.reward
		if self.episode_max_reward is not None:
			self.episode_max_reward = max(self.reward, self.episode_max_reward)
		else:
			self.episode_max_reward = self.reward
		
		if self.levelFinished:
			temp = "___levelFinished" if marioState[0] == 1 else "___marioIsDead"
			print(temp)
			self.save_episode_values(marioState[0])
			self.print_report()			
		
		# Periodicallly print report
		if self.total_iterations % self.report_frequency == 0:
			self.print_report()				

	def getAction(self):
		#print("____getAction")
		if self.old_state is not None:	
			if not self.isSkippedFrame():
				self.chosenAction = self.choose_action(self.old_state)
		else:
			self.firstFrame = False			
			self.chosenAction = self.actions[0]
		#print(self.chosenAction)
		return tuple(self.chosenAction)

	def setObservationDetails(self, rfWidth, rfHeight, egoRow, egoCol):
		#print("____setObservationDetails")
		self.receptiveFieldWidth = rfWidth
		self.receptiveFieldHeight = rfHeight
		self.marioEgoRow = egoRow
		self.marioEgoCol = egoCol
		
	def reset(self):
		print("____reset")
		self.start_new_episode()
	
	def getName(self):
		#print("____getName")
		return self.agentName
	#------------------------------------------------	
	
	#------------------------------------------------ 
	#------- 4 parameters representation (S1) -------
	#------------------------------------------------	
	def computeXa(self):
		oldPosX = 0 if self.oldMarioFloats is None else self.oldMarioFloats[0]
		return self.marioFloats[0] - oldPosX
		
	def computeYa(self):
		oldPosY = 0 if self.oldMarioFloats is None else self.oldMarioFloats[1]	
		return self.marioFloats[1] - oldPosY	

	def computeForwardBlock(self):
		val = 10;
		ypos = self.marioEgoCol
		xpos = self.marioEgoRow
		i = ypos + 1;
		size = self.receptiveFieldWidth
		
		while (i<size):
			if (self.levelScene[xpos][i] == 0 or 
				self.levelScene[xpos][i] == 2 or
				self.levelScene[xpos][i] == 25):
				i += 1
			else:
				val = self.getFloatDistanceToX(i - ypos)
				i = size
		return val	
		
	def getFloatDistanceToX(self, x):
		return x - (self.marioFloats[0] / 16) + int(self.marioFloats[0] / 16)		
	
	def computeBottomBlock(self):
		val = 10;
		ypos = self.marioEgoCol
		xpos = self.marioEgoRow
		i = xpos + 1;
		size = self.receptiveFieldWidth
		
		while (i<size):
			if (self.levelScene[i][ypos] == 0 or 
				self.levelScene[i][ypos] == 2 or
				self.levelScene[i][ypos] == 25):
				i += 1
			else:
			   val = self.getFloatDistanceToY(i - xpos)
			   i = size
		return val

	def getFloatDistanceToY(self, y):
		return y - (self.marioFloats[1] / 16) + int(self.marioFloats[1] / 16)		
	#------------------------------------------------	
