# -*- coding: utf-8 -*-

#Edited by Mathieu in order to be used with Mario AI Benchmark and AmiCoJavaPy

import numpy as np
import tensorflow as tf
#from collections import deque
#import matplotlib.pyplot as plot
import ops
from SarstReplayMemory import SarstReplayMemory

#from marioagent import MarioAgent

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

# Network parameters:
batch_size = 32 # total batch size
gamma = 0.9 # discounted future reward for Q Learning

# LSTM parameters:
useLSTM = True
trace_length = 16 # need to fully divide batch_size
maskHalfLoss = True # Mask the first half loss of every trace of the batch
reset_rnn_state = False # LSTM stateless (True) or stateful (False) (stateless: reset state after every batch, stateful: keep state)
useLSTMTanH = True # Allows to use TanH for LSTM or the activation define in build_network() function 
#  Note: Be careful if activation is non squashing (like ReLU) and LSTM is stateful
#        because batch loss will probably be nan or very high 
#        (clipping gradients isn't working everytime to solve this problem)

# Dueling Network:
useDuelingNetwork = False

# Either number of units in the LSTM or in the fully connected according to "useLSTM" (for stateRepresentationID : 0 or 1)
S0_first_fc_num_units = 512 # also for S2
S1_first_fc_num_units = 256

# saveFiles:
modelCheckpoint = "./modelWeights.ckpt"
hyperparametersFile = "./modelParameters.dqn"
memoryFile = "./modelMemory.dqn"

#-----------------------------

if useLSTM:
	if trace_length > batch_size:
		trace_length = batch_size
	batch_size //= trace_length
else:
	trace_length = 1

if stateRepresentationID == 0 or stateRepresentationID == 2:
	first_fc_num_units = S0_first_fc_num_units
else:
	first_fc_num_units = S1_first_fc_num_units

class MarioDQNAgent():
			 
	"""for this __init__ we need to only have the "self" parameter (cf. AmiCoJavaPy)"""
	def __init__(self):
		print("Initialization of Mario DQN Agent")
		"""print("__networkID = ", networkID)
		print("__rewardID = ", rewardID)
		print("__stateRepresentationID = ", stateRepresentationID)  """		 
		with tf.device('/gpu:0'):
			#with tf.Session() as session: # this line doesn't work when python is embedded	  	
				session = tf.Session()
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
				self.agentName = "MarioDQNAgent"
				self.initAgent(session, state_size, actions, report_frequency)
		self.saver = tf.train.Saver() #tf.all_variables()
		self.restore_model()
		print("Initialized - Agent")
	
	def initAgent(self,
					tf_session,
					state_size,
					actions,
					report_frequency):
		self.init_new_episode()
		
		self.session = tf_session # a tensorflow session
		self.state_size = state_size # size of environment state space.
		self.actions = actions # a list of list of NES buttons, each of which is binary and of length 6
		self.chosenAction = None
		self.num_actions = len(actions) # actions available at any state.
		self.batch_size = batch_size # when updating deep q network, how many sarst samples to randomly pull
		self.trace_length = trace_length
		self.gamma = gamma
		self.cur_state = None
		self.old_state = None
		self.firstFrame = True
		
		#----------Other parameters----------
		
		# Frequencies
		self.compute_action_frequency = 1 # Step before computing another action (4 is often used but isn't giving good results here)
		self.target_network_update_frequency = 4000 / self.compute_action_frequency # how many minibatch steps before we set the target network weights to the prediction network weights
		self.save_model_frequency = 2000 / self.compute_action_frequency
	
		# The size of the SarstReplayMemory class
		self.memory_capacity = 100000

		# All learning rate configuration goes here
		self.learning_rate_init = 0.00025 # 0.00025 used in Deep Mind atari paper
		self.learning_rate_decay = 0.99999
		self.learning_rate_decay_steps = 5
		self.learning_rate_min = 0.0001

		# and some exploration parameters		
		self.epsilon = 1.0				
		self.epsilon_min = 0.05
		self.epsilon_decay = 0.9999
	
		# we wont clip the loss, but we will clip gradients themselves.
		self.clip_gradients_enabled = False
		self.clip_gradients_by_global_norm = True # otherwise clip with min max
		self.norm_gradient = 5.0
		self.min_gradient = -10.
		self.max_gradient = 10. # gradients of more than this will be clipped to this maximum in optimizer's minimize() call
	
		# use dropout on the weights
		self.dropout_keep_probability = 1.0 # layer 1 and 2 outputs pass through dropout
		
		#------------------------------------	
		
		# The Deep Network will have a prediction network and a target network.
		self.network_inputs = {}
		self.rnn_trainLength = {}
		self.rnn_batch_size = {}
		self.rnn_state_in = {}
		self.rnn_state = {}
		self.rnn_state_train = {}		
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
		self.episode_q_min = None
		self.episode_q_max = None
		self.episode_q_total = 0		
	
		# How often we get an update printed to screen about how things are looking and what param values are
		self.report_frequency = report_frequency
	
		# instantiate a new blank replay memory to store state, action, reward, new_state, new_state_is_terminal arrays
		self.replay_memory = SarstReplayMemory(self.memory_capacity,
											   self.state_size,
											   useLSTM,
											   self.trace_length)
		print("Initialized - SARST Replay Memory")
		
		# Let's also make our agent a prediction network, and a target network
		# construct the deep q network, which initializes all the placeholders for all variables in the tf graph
		self.deep_q_network()
		print("Initialized - Deep Q Network")		
		
	#------------------------------------------------
	#---------------- Neural Network ----------------
	#------------------------------------------------		
	def deep_q_network(self):
		# Construct the two networks with identical architectures
		self.build_network('prediction_network')
		self.build_network('target_network', prediction_network=False)

		# Init rnn state
		if useLSTM:
			self.rnn_state_train['prediction_network'] = (np.zeros([self.batch_size,first_fc_num_units]), np.zeros([self.batch_size,first_fc_num_units])) 		
			self.rnn_state_train['target_network'] = (np.zeros([self.batch_size,first_fc_num_units]), np.zeros([self.batch_size,first_fc_num_units])) 		
		
		# Create tensorflow placeholder variables and map functions that we can call in the session
		# to easily copy over the prediction network parameters to the target network parameters
		self.build_network_copier()
		
		# Build global step decayer for the learning rate exponential decay function
		with tf.variable_scope('global_step'):
			self.global_step = tf.placeholder(tf.int32, name="global_step")
		
		# create the optimizer in the model
		self.run_optimizer()
		
		# initialize all these variables, mostly with xavier initializers
		init_op = tf.global_variables_initializer()
		
		# Ready to train
		self.session.run(init_op)	
	
	# This function will be used to build both the prediction network as well as the target network
	def build_network(self, scope_name, prediction_network=True):
		activation = tf.nn.leaky_relu
		#activation = tf.nn.relu
		net_shape = [None] + [s for s in self.state_size]
		print(net_shape)
		
		with tf.variable_scope(scope_name):
			self.network_inputs[scope_name] = tf.placeholder(tf.float32, shape=net_shape, name=scope_name+"_inputs")
			
			#----- Deep Q Network (with convolution) -----
			if stateRepresentationID == 0 or stateRepresentationID == 2:
				# 16 9*9 filters with stride 4
				conv1 = ops.conv(self.network_inputs[scope_name],
						16,
						kernel=[9,9],
						strides=[4,4],
						w_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
						name="conv1")
				conv1 = activation(conv1)

				# 32 5*5 filters with stride 2
				conv2 = ops.conv(conv1,
				#conv2 = ops.conv(self.network_inputs[scope_name],			
						32,
						kernel=[5,5],
						strides=[2,2],
						w_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
						name="conv2")
				conv2 = activation(conv2)

				# 64 3*3 filters with stride 1
				conv3 = ops.conv(conv2,
						64,
						kernel=[3,3],
						strides=[1,1],
						w_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
						name="conv3")
				conv3 = activation(conv3)

				conv3 = tf.contrib.layers.flatten(conv3)
				num_units = first_fc_num_units
				# First fully-connected layer (LSTM or not)	
				fc1_out = self.build_before_last_fc_layer(conv3, num_units, scope_name, activation)

				# Final layer (Dueling or not)
				self.build_last_fc_layer(fc1_out, num_units, scope_name, prediction_network)
					
			# ----- Q Network with x hidden layer -----
			elif stateRepresentationID == 1:
				#inputLayer = tf.contrib.layers.flatten(self.network_inputs[scope_name])
				inputLayer = self.network_inputs[scope_name]
				
				"""# fully-connected layer
				num_units = 128
				with tf.variable_scope('fc1') as scope:
					w1 = tf.get_variable('fcw1', shape=[inputLayer.get_shape()[1], num_units],
							initializer=tf.contrib.layers.xavier_initializer())
					b1 = tf.get_variable('fcb1', shape=[num_units],
							initializer=tf.constant_initializer(0.0))
					fc0_out = tf.matmul(inputLayer, w1) + b1
					fc0_out = activation(fc0_out)
					fc0_out = tf.nn.dropout(fc0_out, keep_prob=self.dropout_keep_probability)
				inputLayer = fc0_out"""
				
				"""# fully-connected layer
				num_units = 256
				with tf.variable_scope('fc3') as scope:
					w3 = tf.get_variable('fcw3', shape=[inputLayer.get_shape()[1], num_units],
							initializer=tf.contrib.layers.xavier_initializer())
					b3 = tf.get_variable('fcb3', shape=[num_units],
							initializer=tf.constant_initializer(0.0))
					fc3_out = tf.matmul(inputLayer, w3) + b3
					fc3_out = activation(fc3_out)
					fc3_out = tf.nn.dropout(fc3_out, keep_prob=self.dropout_keep_probability)
				inputLayer = fc3_out"""
				
				num_units = first_fc_num_units
				# fully-connected layer (LSTM or not)	
				fc1_out = self.build_before_last_fc_layer(inputLayer, num_units, scope_name, activation)

				# Final layer (Dueling or not)
				self.build_last_fc_layer(fc1_out, num_units, scope_name, prediction_network)
					
	# ----- Build before last fc layer ----- (LSTM or not)					
	def build_before_last_fc_layer(self, inputs, num_units, network_scope_name, activation):
		"""if stateRepresentationID == 2:
		self.additionnal_inputs = tf.placeholder(tf.float32, shape=net_shape, name=scope_name+"additionnal_inputs")
		inputs = tf.concat([inputs, self.additionnal_inputs], 0)"""	
		if not useLSTM:
			# Fully-connected layer
			with tf.variable_scope('fc1') as scope:
				w1 = tf.get_variable('fcw1', shape=[inputs.get_shape()[1], num_units],
						initializer=tf.contrib.layers.xavier_initializer())
				b1 = tf.get_variable('fcb1', shape=[num_units],
						initializer=tf.constant_initializer(0.0))
				fc1_out = tf.matmul(inputs, w1) + b1
				fc1_out = activation(fc1_out)
				fc1_out = tf.nn.dropout(fc1_out, keep_prob=self.dropout_keep_probability)
		else:
			# LSTM layer		
			with tf.variable_scope('lstm') as scope:
				if useLSTMTanH:
					rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=num_units)
				else:
					rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=num_units, activation=activation)
				self.rnn_trainLength[network_scope_name] = tf.placeholder(dtype=tf.int32)
				# We take the output from the final convolutional layer and send it to a recurrent layer.
				# The input must be reshaped into [batch x trace x units] for rnn processing, 
				# and then returned to [batch x units] when sent through the upper levels.
				self.rnn_batch_size[network_scope_name] = tf.placeholder(dtype=tf.int32, shape=[])
				rnn_convFlat = tf.reshape(inputs,[self.rnn_batch_size[network_scope_name], self.rnn_trainLength[network_scope_name], inputs.get_shape()[1]])
				self.rnn_state_in[network_scope_name] = rnn_cell.zero_state(self.rnn_batch_size[network_scope_name], tf.float32)
				rnn, self.rnn_state[network_scope_name] = tf.nn.dynamic_rnn(\
					inputs=rnn_convFlat, cell=rnn_cell, dtype=tf.float32, initial_state=self.rnn_state_in[network_scope_name])
				rnn = tf.reshape(rnn, shape=[-1,num_units])
				fc1_out = rnn
				
		return fc1_out	
	
	# ----- Build last fc layer ----- (Dueling or not)
	def build_last_fc_layer(self, inputs, num_units, network_scope_name, prediction_network):
		if not useDuelingNetwork:					
			# Final fully-connected layer
			with tf.variable_scope('fc2') as scope:
				w2 = tf.get_variable('fcw2', shape=[num_units, self.num_actions],
						initializer=tf.contrib.layers.xavier_initializer())
				b2 = tf.get_variable('fcb2', shape=[self.num_actions],
						initializer=tf.constant_initializer(0.0))
				if prediction_network:
					self.q_predictions = tf.matmul(inputs, w2) + b2				
					self.max_predict_q_action = tf.argmax(self.q_predictions, 1)
				else:
					self.q_targets = tf.matmul(inputs, w2) + b2		
		else:
			# Dueling Network
			with tf.variable_scope('fc2') as scope:
				streamA, streamV = tf.split(inputs, 2, 1)
				#AW = tf.Variable(tf.random_normal([num_units//2, self.num_actions]))
				#VW = tf.Variable(tf.random_normal([num_units//2, 1]))
				AW = tf.get_variable('aw', shape=[num_units//2, self.num_actions], initializer=tf.contrib.layers.xavier_initializer())
				VW = tf.get_variable('vw', shape=[num_units//2, 1], initializer=tf.contrib.layers.xavier_initializer())				
				Advantage = tf.matmul(streamA, AW)
				Value = tf.matmul(streamV, VW)
				salience = tf.gradients(Advantage, self.network_inputs[network_scope_name])
				# Then combine them together to get our final Q-values.
				if prediction_network:
					self.q_predictions = Value + tf.subtract(Advantage, tf.reduce_mean(Advantage, reduction_indices=1, keepdims=True))
					self.max_predict_q_action = tf.argmax(self.q_predictions, 1)
				else:
					self.q_targets = Value + tf.subtract(Advantage, tf.reduce_mean(Advantage, reduction_indices=1, keepdims=True))

	def build_network_copier(self):
		# Tensorflow needs to create copy operations for parameters for which we can call eval()
		# When we call eval, we can pass a feed dictionary of the copied parameters
		with tf.variable_scope('copy_weights'):
			self.copied_parameters = {}
			self.copy_parameters_operation = {}
		for idx, (predict_parameter, target_parameter) in enumerate(zip(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='prediction_network'),
																	  tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_network'))):
			# sanity check that parameters are the same shape
			assert predict_parameter.get_shape().as_list() == target_parameter.get_shape().as_list(), "Networks parameters must be the same shape"
			input_shape = predict_parameter.get_shape().as_list()
			self.copied_parameters[predict_parameter.name] = tf.placeholder(tf.float32, shape=input_shape, name="copier_%d"%idx)
			self.copy_parameters_operation[predict_parameter.name] = target_parameter.assign(self.copied_parameters[predict_parameter.name])
		
	def copy_prediction_parameters_to_target_network(self):
		print("\nCopying prediction network parameters to target network")
		for param in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='prediction_network'):
			self.copy_parameters_operation[param.name].eval({ self.copied_parameters[param.name] : param.eval(session = self.session) }, session = self.session)	
	
	def clip_gradients(self, gradient):
		# Clip gradients to min_grad and max_grad before.
		# Used before optimizer.minimize() function
		# 
		# Parameters
		# gradient (int) - single value from gradient list returned from a tuple of (gradient, variable) 
		#			   - from a call to tf.optimizer.compute_gradients(loss_function)
		# min_grad, max_grad (float) - min and max gradients
		#
		# Returns
		# - tensorflow gradient after being clipped
		#
		if gradient is None:
			return gradient #this is necessary for initialization in tensorflow or it throws an error
		return tf.clip_by_value(gradient, self.min_gradient, self.max_gradient)
	
	# ----- Optimizer -----
	def run_optimizer(self):
		with tf.variable_scope('optimizer'):
			# target y variables for use in the loss function in algorithm 1
			self.target_y = tf.placeholder(tf.float32, shape=[None], name="target_y")
			
			# chosen actions for use in the loss function in algorithm 1
			self.chosen_actions = tf.placeholder(tf.int32, shape=[None], name="chosen_actions")
			
			# convert the chosen actions to a one-hot vector.
			self.chosen_actions_one_hot = tf.one_hot(self.chosen_actions,
												   self.num_actions,
												   on_value=1.0,
												   off_value=0.0,
												   axis=None,
												   dtype=None,
												   name="chosen_actions_one_hot")
			
			# The q value is that of the dot product of the prediction network
			# with the one-hot representation of the chosen action. this gives us a single chosen action
			# because reduce_sum will add this up over each of the indexes, all but one of which are non-zero
			self.predict_y = tf.reduce_sum(self.chosen_actions_one_hot * self.q_predictions,
										 axis=1, #reduce along the second axis because we have batches
										 name="predict_y")
			
			# Loss - Implement mean squared error between the target and prediction networks as the loss
			#self.loss = tf.reduce_mean(tf.square(tf.subtract(self.target_y, self.predict_y)), name="loss")
			#self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.predict_y, logits=self.target_y), name="loss")			
			self.loss = tf.square(tf.subtract(self.target_y, self.predict_y))
			#self.loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.predict_y, logits=self.target_y)
			if useLSTM and maskHalfLoss:
				self.maskA = tf.zeros([self.batch_size,self.trace_length//2])
				self.maskB = tf.ones([self.batch_size,self.trace_length//2])
				self.mask = tf.concat([self.maskA,self.maskB],1)
				self.mask = tf.reshape(self.mask,[-1])
				self.loss = self.loss * self.mask #self.loss = tf.reduce_mean(self.loss * self.mask)
			self.loss = tf.reduce_mean(self.loss, name="loss")
			
			# Decay learning rate
			self.learning_rate = tf.maximum(self.learning_rate_min, 
											tf.train.exponential_decay(self.learning_rate_init,
																	self.global_step,
																	self.learning_rate_decay_steps,
																	self.learning_rate_decay))

			# and pass it to the optimizer to train on this defined loss function
			#self.opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate, name="adam")
			self.opt = tf.train.RMSPropOptimizer(self.learning_rate, momentum=0.95, epsilon=0.01)
			#self.opt = tf.train.RMSPropOptimizer(self.learning_rate, 0.99, 0.0, 1e-6)
			
			# Clip the gradients if the option is enabled by breaking the optimizer's minimize() into a (get, clip, apply) set of operations
			if self.clip_gradients_enabled:
				gradients_and_variables = self.opt.compute_gradients(self.loss)
				if self.clip_gradients_by_global_norm:
					gradients, variables = zip(*gradients_and_variables)
					gradients, _ = tf.clip_by_global_norm(gradients, self.norm_gradient)
					clipped_gradients = zip(gradients, variables)	
				else:
					clipped_gradients = [(self.clip_gradients(grad), var) for grad, var in gradients_and_variables]
				self.optimizer = self.opt.apply_gradients(clipped_gradients) # this increments global step
			else:
				self.optimizer = self.opt.minimize(self.loss)
	
	# ----- Minibatch -----			
	def run_minibatch(self):
		#print("Running minibatch. Sarst memory contains %d entries" % self.replay_memory.memory_size)
		
		# These are of size batch_size, not single values
		state, action, reward, state_prime, state_prime_is_terminal = self.replay_memory.get_batch_sample(self.batch_size)
		
		# TODO - Split state and state_prime to use stateRepresentationID == 2 and feed them correctly to the network
		#if stateRepresentationID == 2:

		# get the logits from the target network for this resulting state
		if not useLSTM:
			q_value_state_prime = self.q_targets.eval({self.network_inputs['target_network'] : state_prime}, session = self.session)
		else:
			if reset_rnn_state:
				state_train = (np.zeros([self.batch_size,first_fc_num_units]), np.zeros([self.batch_size,first_fc_num_units])) 
			else:
				state_train = self.rnn_state_train['target_network']		
			#q_value_state_prime = self.q_targets.eval({...}, session = self.session)
			q_value_state_prime, self.rnn_state_train['target_network'] = self.session.run(
					[self.q_targets, self.rnn_state['target_network']], 
					{self.network_inputs['target_network'] : state_prime,
					self.rnn_trainLength['target_network'] : self.trace_length,
					self.rnn_batch_size['target_network'] : self.batch_size,
					self.rnn_state_in['target_network'] : state_train})
		
		# the max logit is the max action q value
		max_q_value_state_prime = np.max(q_value_state_prime, axis=1)
		
		# the state_prime_is_terminal * 1  converts [True, False, True] to [1,0,1].
		# Subtracting this from 1 effectively eliminates the entire term, leaving just reward for terminal states
		target_y = reward + (self.gamma * max_q_value_state_prime * (1 - (state_prime_is_terminal*1)) )
		
		# Now that the terms are in place, run a session
		if not useLSTM:
			_, self.report_predictions, lr, self.one_hot_actions, self.final_predictions, self.report_loss = self.session.run(
					[self.optimizer, self.q_predictions, self.learning_rate, self.chosen_actions_one_hot, self.predict_y, self.loss],
					{self.network_inputs['prediction_network'] : state,	 # it'll need the states possibly
					self.chosen_actions : action, # and definitely the actions
					self.target_y : target_y,	 # and the targets in the optimizer
					self.global_step : self.total_iterations # and update our global step. TODO, maybe this should be self.global_step. make sure isn't incremented twice with minimize() call
				})
		else:
			if not reset_rnn_state:
				state_train = self.rnn_state_train['prediction_network']
			_, self.report_predictions, lr, self.one_hot_actions, self.final_predictions, self.report_loss, self.rnn_state_train['prediction_network'] = self.session.run(
					[self.optimizer, self.q_predictions, self.learning_rate, self.chosen_actions_one_hot, self.predict_y, self.loss, self.rnn_state['prediction_network']],
					{self.network_inputs['prediction_network'] : state,	 # it'll need the states possibly
					self.chosen_actions : action, # and definitely the actions
					self.target_y : target_y,	 # and the targets in the optimizer
					self.global_step : self.total_iterations, # and update our global step. TODO, maybe this should be self.global_step. make sure isn't incremented twice with minimize() call
					self.rnn_trainLength['prediction_network'] : self.trace_length,
					self.rnn_batch_size['prediction_network'] : self.batch_size,
					self.rnn_state_in['prediction_network'] : state_train})
		
		#if self.episode_iterations == 0 and self.episode_count % 5 == 0:
		#	print "Episode %d\t\tLearning Rate: %.9f\t\tEpsilon: %.6f" % (self.episode_count, lr, self.epsilon)
		if self.total_iterations % self.report_frequency == 0:		
			print("lr=",lr)
		
		self.minibatches_run += 1
		
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
			# returns a tensor with the single best q action evaluated from prediction network
			# [0] returns this single value from the tensor
			# max_predict_q_action					
			if not useLSTM:
				aq = self.q_predictions.eval({self.network_inputs['prediction_network'] : [state]}, session = self.session) #[0]
			else:
				if reset_rnn_state:
					rnn_state = (np.zeros([1,first_fc_num_units]), np.zeros([1,first_fc_num_units]))
				else:
					rnn_state = self.rnn_state_train['prediction_network']
					rnn_state = (np.array([rnn_state[0][-1]]), np.array([rnn_state[1][-1]]))
				aq = self.q_predictions.eval({self.network_inputs['prediction_network'] : [state],
						self.rnn_trainLength['prediction_network'] : 1,
						self.rnn_batch_size['prediction_network'] : 1,
						self.rnn_state_in['prediction_network'] : rnn_state}, session = self.session) #[0]
			
			a = np.argmax(aq, axis=1)[0]
			q = aq[0][a]
			
			# Update Q metrics		
			self.episode_q_total += q											
			if self.episode_q_min is not None:
				self.episode_q_min = min(q, self.episode_q_min)
			else:
				self.episode_q_min = q
			if self.episode_q_max is not None:
				self.episode_q_max = max(q, self.episode_q_max)
			else:
				self.episode_q_max = q
											
			return self.actions[a]	
			
	def isSkippedFrame(self):
		return self.episode_iterations % self.compute_action_frequency != 0
		#return np.random.rand() < 1. / self.compute_action_frequency
	#------------------------------------------------								
		
	#------------------------------------------------
	#---------------- Save & Restore ----------------
	#------------------------------------------------
	def save_model(self):
		# Save model
		self.saver.save(self.session, modelCheckpoint)
		print("=== Model saved as \"{}\" ===".format(modelCheckpoint))
		# Save memory
		self.save_memory(memoryFile)	
		# Save parameters
		np.savez(hyperparametersFile, 
			epsilon = self.epsilon, 
			total_iterations = self.total_iterations, 
			total_random_action = self.total_random_action)		
		print("=== Parameters saved as \"{}.npz\" ===\n".format(hyperparametersFile))	
		
	def save_memory(self, memoryFile):	
		self.replay_memory.save_memory(memoryFile)
			  
	def restore_model(self):
		if(tf.train.checkpoint_exists("checkpoint")):
			# Restore model
			self.saver.restore(self.session, modelCheckpoint)
			print("\n=== Model restored from \"{}\" ===".format(modelCheckpoint))
			# Restore memory
			self.restore_memory(memoryFile)			
			# Restore parameters
			npzFile = np.load("{}.npz".format(hyperparametersFile))
			self.epsilon = npzFile['epsilon']
			self.total_iterations = npzFile['total_iterations']
			self.total_random_action = npzFile['total_random_action']			
			print("=== Parameters restored from \"{}.npz\" ===\n".format(hyperparametersFile))
		else:
			self.copy_prediction_parameters_to_target_network()
			
	def restore_memory(self, memoryFile):	
		self.replay_memory.restore_memory(memoryFile)
		
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
											self.episode_ennemy_killed,
											self.episode_q_total,
											self.episode_q_min,
											self.episode_q_max,
											self.episode_q_total / self.episode_iterations)
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
		self.episode_q_min = None
		self.episode_q_max = None			
		self.episode_q_total = 0			
		self.init_new_episode()							
						
	def print_report(self):
		print("(",self.total_random_action,"/",self.total_iterations,") _ epsilon =",self.epsilon)
		print("reward =",self.reward,"_ total_reward =",self.total_episode_reward)
		print("memory_size = ",self.replay_memory.memory_size,"_ Batch Loss: %.4f" % self.report_loss)
		
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
		
		# Then we store away what happened, unless we are in the first stage
		if self.old_state is not None:
			self.replay_memory.add_to_memory(self.old_state,
											self.actions.index(self.chosenAction), # SARST has no access to button maps
											self.reward,
											self.cur_state,
											self.levelFinished)

		self.old_state = self.cur_state

		# Run an actual neural network trainer on that replay memory
		if self.replay_memory.memory_size >= self.batch_size * self.trace_length:
			self.run_minibatch()	
		
		# Increment counters for decayed variables
		self.total_iterations += 1	
		
		# Decay agent exploration 
		self.epsilon = max(self.epsilon_min, self.epsilon*self.epsilon_decay)
		
		# Update the target network, however often you choose to
		if self.total_iterations % self.target_network_update_frequency == 0:
			self.copy_prediction_parameters_to_target_network()
		
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
		
		# Periodically save the model		
		if self.total_iterations % self.save_model_frequency == 0:
			self.save_model()
		
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
