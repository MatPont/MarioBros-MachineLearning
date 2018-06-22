Hello, this is the directory for the MarioAI Benchmark and agents.

## Agents

You'll find the python agents in the directory: [`/src/main/java/amico/python/agents`](https://github.com/MatPont/MarioBros-MachineLearning/tree/master/mario-ai-master/src/main/java/amico/python/agents).

* Deep Q Learning is in `MarioDQNAgent2.py` it needs:
  * Tensorflow (I used tensorflow-gpu 1.5.0 with CUDA 9.0.176 and cuDNN 7.0.5).
  * Numpy (I used 1.14.3).
  * The files: ops.py, SarstReplayMemory.py and PrioritizedSumTree.py contained in the same directory than the agent.
* NEAT is in `MarioNEATAgent.py` and is not fully implemented yet.
* NEAT + Q is not implemented yet.

## Run Agents

A library must be created to make possible communication between the agent in Python and the MarioAI Benchmark in Java. 

Some files are given with the benchmark to make easier this library, it was first for python 2 so I modified the files to use python 3. I tried everything in Linux, therefore the process for Windows and Mac (darwin) aren't described here. Good luck if you use these OS. 

You'll probably need to configure JavaPy or PyJava, see below.

In this directory run the script: 
* `runDQN.sh` to run DQN Agent
* `runNEAT.sh` to run NEAT Agent
* `runNEATQ.sh` to run NEAT + Q Agent

I ran experiments on my laptop and on a slurm client called OSIRIM, the process will probably not be the same but if you're planing to use a slurm client look at `README_OSIRIM.md`.

### Configure Deep Q Learning

I used JavaPy (which means that Java calls Python) for Deep Q learning.

If you use Linux you'll probably need to modify in the directory [`/src/main/java/amico/python/JavaPy`](https://github.com/MatPont/MarioBros-MachineLearning/tree/master/mario-ai-master/src/main/java/amico/python/JavaPy) some files:
* in `Makefile.Linux` the variables:
  * `JAVADIRPATH` must be initialized with your java directory.
  * `PYTHONVERSION` must be initalized with your version of python.
* in `src/ch_idsia_tools_amico_AmiCoJavaPy.cc` the variable `pythonLibName` should be initialized with the library name of your python, for me it's `libpython3.5m.so` and you'll probably need to only replace "3.5" with your python version.

In the `__init__` function of the MarioDQNAgent class you need to specify the tensorflow device in the line `with tf.device('/gpu:0'):`, if you're not using tensorflow-gpu replace `gpu` by `cpu`

#### Custom Mario environment

You can configure the mario environment in the Main class of [`src/main/java/ch/idsia/scenarios`](https://github.com/MatPont/MarioBros-MachineLearning/tree/master/mario-ai-master/src/main/java/ch/idsia/scenarios). This Main class define the training environment of the agent, like the number of levels before the end of training, their difficulty, the time limit for a level etc. It's mainly the variable `marioAIOptions` that will allow you to custom the environment.

### Configure NEAT and NEAT + Q

I used PyJava (which means that Python calls Java) for NEAT and probably for NEAT + Q.

## Configure Agents

We used two state representations:
* (**S0**) Grid view: a grid of size 19*19 where Mario is always in the center. Each cell contains a number representing a coin, an object, an enemy etc. It is a simplified view from the pixels view.
* (**S1**) 4 parameters view: speed of Mario on X and Y axis and distance of the first obstacle to the right and the bottom of Mario.

and two reward functions:
* (**R0**) Score: reward related to the score of the game.
* (**R1**) Right and top: the agent is rewarded when he goes to the right and the top.

To change state representation and reward function you need to modify the variables `stateRepresentationID` and `rewardID` at the beginning of an agent file.

* DQN Agent:
  * `useLSTM`: if True the before last layer will be LSTM
    * `trace_length`: length of a memory when it is took from the replay memory.
    * `maskHalfLoss`: if True it will mask (reset to 0) the first half of each trace [Hausknecht et al. 2015].
    * `reset_rnn_state`: if True LSTM will be stateless (meaning that LSTM state is reset after every batch) otherwise it will be stateful (state is keep for the next batch).
    * `useLSTMTanH`: if True LSTM use TanH for its inner activation function, otherwise it will use the activation function define in `build_network()` function (it's leaky_relu)
  * `useDuelingNetwork`: if True the last layer will be dueling [Wang et al. 2016].
  * You have also all the basic parameters like batch size, gamma (discouted reward factor), initial epsilon (exploration rate), minimum epsilon and epsilon decay, replay memory capacity, frequency of computing an action, frequency of updating target network and another one for saving model periodically.
  * Learning rate decay exponentially, you have the parameters like initial learning rate, learning rate decay, steps and minimum learning rate.
  * You can clip gradients either with global norm or with min max clipping.
  * The optimizer is RMSProp, there is also in comment an Adam optimizer.
  * You can either compute loss with mean squared error or with "softmax cross entropy with logits" that is commented in the code.
  * With DQN we have two networks:
    * for **S0**: it's a convolutional neural network, with 3 convolutional layer and 2 fully-connected layer.
    * for **S1**: it's a basic neural network with 1 hidden layer.

* NEAT Agent:
  * .
  
## Analyze results

A python script called `resultManager.py` can help you to analyse the results of your agent (the file containing results is in `src/main/bin/AmiCoBuild/{JavaPy|PyJava}/episode_values.txt`).
