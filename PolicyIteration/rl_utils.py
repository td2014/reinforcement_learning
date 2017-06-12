#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 05:34:57 2017

Code to do policy iteration for reinforcement learning.

@author: anthonydaniell
"""
import numpy as np
import sys

#
# Create value lookup function
#
def valLookup(grid, direction, refPosition, deltaState=None):
    
    if deltaState==None: # use a default
        deltaState={'l': np.array([0,-1]), 
                    'r': np.array([0,1]),
                    'u': np.array([-1,0]),
                    'd': np.array([1,0])}
    
    try:
        testPosition=refPosition+deltaState[direction] # update position
        
        if testPosition[0]<0:
            testPosition=refPosition  # prevent moving off world boundary
    
        if testPosition[0]>=grid.shape[0]:
            testPosition=refPosition  # prevent moving off world boundary

        if testPosition[1]<0:
            testPosition=refPosition  # prevent moving off world boundary

        if testPosition[1]>=grid.shape[1]:
            testPosition=refPosition  # prevent moving off world boundary

    except: # error
        sys.exit('Please enter a valid direction.') 
    
    # normal exit
    return grid[testPosition[0],testPosition[1]]


#
# Return policy
#

def policyLookup(policy, state, actions):
        
    estLen=100 # ccreate vector for random sampling
    actionList=[]
    currPolicyGivenState=policy[str(state)]
    for iAction in actions:
        actionCount=int(currPolicyGivenState[iAction]*1.0*estLen)
        for j in range(actionCount):
            actionList.append(iAction)
        
    while len(actionList)<estLen: # make sure we are not empty
        actionList.append(iAction)
        
    actionList=actionList[:estLen] # clip output
        
    return actionList[np.random.randint(0,estLen)]

#
# End of script
#