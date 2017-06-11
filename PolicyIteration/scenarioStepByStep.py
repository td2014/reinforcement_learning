#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 09:21:00 2017

Driver structure to evaluate value of an implicit MDP

@author: anthonydaniell
"""
import numpy as np
np.random.seed(42)
import rl_utils as rlut
#
# Create grid world
#
gridRows=1
gridCols=7
termLocations = []
stepReward=0  # cost to move one grid location
gridWorld = np.zeros([gridRows,gridCols])

actions=['l','r']
deltaState={'l': np.array([0,-1]), 
            'r': np.array([0,1])}
                
#
# Initialize values
#
termLocations.append(np.array([0,0]))  # set first grid element to be terminal node
termLocations.append(np.array([0,6]))  # set last grid element to be terminal node
#
#
print('main: initial gridworld:')
print(gridWorld)
print()
print('main: initial gridworld - termLocations:')
print(termLocations)
print()

#
# Choose an initial state
#

initState=np.array([0,3])

#
# Choose a policy
#
# action = F(state)
#
policy=None

#
# Do a monte carlo run
#
numEpisodes=100
for iEpisode in range(numEpisodes):
    print('------')
    print('main: episode = ', iEpisode)
    print('main: initState = ', initState)
    print()
    termState=False
    refPosition=initState
    gridStateCounter = np.zeros([gridRows,gridCols])
    while not termState:
# loop
#  --Get move from policy
        if policy==None: # random action
            # do updates
            gridStateCounter[refPosition[0],refPosition[1]]= \
                gridStateCounter[refPosition[0],refPosition[1]]+1
            # Look ahead
            currAction = actions[np.random.randint(0,len(actions))]
            currVal = 0 #rlut.valLookup(gridWorld, currAction, refPosition, deltaState)
           
            # special case of final location reward.  Should be a
            # function of location and action in general.
            if np.array_equal(refPosition,np.array([0,5])) and currAction=='r':
                currReward=1
            else:
                currReward=0
                
            gridWorld[initState[0], initState[1]]=currReward+gridWorld[initState[0], initState[1]]
            # Check for terminal state
            for iTerm in termLocations:  # check for terminal state
                if np.array_equal(refPosition,iTerm):
                    termState=True
                    
            newPosition=refPosition+deltaState[currAction]
            refPosition=newPosition
            print('main: currAction = ', currAction)
            print('main: refPosition = ', refPosition)
            print('main: currVal = ', currVal)
            print('main: gridWorld = ', gridWorld)
            print()
            
#  --Get reward and add to sum
# Update value when term state is reached
    print('main: gridStateCounter = ', gridStateCounter)
###    flatCount=np.ndarray.flatten(gridStateCounter)
###    for iCount in range(len(flatCount)):
###        flatCount[iCount]=max(flatCount[iCount],1)
        
###    gridStateCounter=np.reshape(flatCount,gridStateCounter.shape)
    
###    print('main: gridStateCounter after update = ', gridStateCounter)
        
    # update values
###    gridWorld=np.divide(gridWorld,gridStateCounter)
    print('main: gridWorld values after updates: ', gridWorld)

#
# Do a TD run
#
####while not termState:
# loop
# --get move from policy
# --update value
#
####    pass

#
# Report out value estimates from MC and TD approaches
#    


#
# End of script
#