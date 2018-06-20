# -*- coding: utf-8 -*-
__author__ = "Sergey Karakovskiy, sergey at idsia fullstop ch"
__date__ = "$${date} ${time}$"

import sys
import os

from forwardjumpingagent import ForwardJumpingAgent
import numpy as np

from evaluationinfo import EvaluationInfo

from PyJavaInit import amiCoSimulator

if __name__ == "__main__":
	libamico, reset, getEntireObservation, performAction, getEvaluationInfo, getObservationDetails, options = amiCoSimulator()
	
	agent = ForwardJumpingAgent()

	options = ""
	if len(sys.argv) > 1:
		options = sys.argv[1]

	if options.startswith('"') and options.endswith('"'):
		options = options[1:-1]

	k = 1
	seed = 0
	print("Py: ======Evaluation STARTED======")
	totalIterations = 0
	for i in range(k, k+10000):
		options1 = options + " -ls " + str(seed)
		print("options: ", options1)
		reset(options1.encode('utf-8'))
		obsDetails = getObservationDetails()
		agent.setObservationDetails(obsDetails[0], obsDetails[1], obsDetails[2], obsDetails[3])
		while (not libamico.isLevelFinished()):
			totalIterations +=1 
			libamico.tick();
			obs = getEntireObservation(1, 0)

			agent.integrateObservation(obs[0], obs[1], obs[2], obs[3], obs[4]);
			action = agent.getAction()
			#print("action: ", action)
			performAction(action);
		print("Py: TOTAL ITERATIONS: ", totalIterations)
		#evaluationInfo = getEvaluationInfo()	
		#print("evaluationInfo = \n", EvaluationInfo(evaluationInfo))
		seed += 1


