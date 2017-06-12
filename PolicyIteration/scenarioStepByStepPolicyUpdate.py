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

initState=np.array([0,1])

#
# Choose a policy
#
# Create random policy to initialize
# and Q
#
newPolicy={}
newQ = {}
for iRow in range(gridRows):
    for iCol in range(gridCols):
        pAct = 1.0/len(actions) # start with uniform random actions
        pQ = 0.0 # initialize Q value
        tmpPolicy = {}
        tmpQ = {}
        for iAct in range(len(actions)):
            tempState = np.array([iRow,iCol])
            tempAct = actions[iAct]
            tmpPolicy[tempAct] = pAct
            tmpQ[tempAct] = pQ
        newPolicy[str(tempState)] = tmpPolicy
        newQ[str(tempState)]=tmpQ
        
policy = newPolicy           
#
# Do a monte carlo run
#
numEpisodes=10
for iEpisode in range(numEpisodes):
    print('------')
    print('main: episode = ', iEpisode)
    print('main: initState = ', initState)
    print()
    termState=False
    refPosition=initState
    sequenceHistory = {}
    finalReward=0
    
    # loop
    while not termState:
#  --Get move from policy
        if policy==None: # random action
            currAction = actions[np.random.randint(0,len(actions))]
        else:
            currAction = rlut.policyLookup(policy, refPosition, actions)
        
        print('main: refPosition = ', refPosition)
        print('main: action = ', currAction)
        # do updates
        try:  # Have we seen this state before?
            tmpStateAction = sequenceHistory[str(refPosition)]
            try:  #Have we seen this action before?
                tmpStateAction[currAction]=tmpStateAction[currAction]+1
            except:
                tmpStateAction[currAction] = 1
            sequenceHistory[str(refPosition)]= tmpStateAction
        except:
            sequenceHistory[str(refPosition)] = {currAction:1}    
        
        # Look ahead
        currVal = 0 #rlut.valLookup(gridWorld, currAction, refPosition, deltaState)
           
        # special case of final location reward.  Should be a
        # function of location and action in general.
        if np.array_equal(refPosition,np.array([0,5])) and currAction=='r':
            currReward=1
        else:
            currReward=0
         
        #update state
        finalReward=finalReward+currReward
###        gridWorld[initState[0], initState[1]]=currReward+gridWorld[initState[0], initState[1]]
        
        refPosition=refPosition+deltaState[currAction]
        # Check for terminal state
        for iTerm in termLocations:
            if np.array_equal(refPosition,iTerm):
                termState=True
                print('TermState True.')
                    
        print('main: refPosition (after update) = ', refPosition)
        print('main: currVal = ', currVal)
        print('main: currReward = ', currReward)
        if termState:
            break
###        print('main: gridWorld = ', gridWorld)
        print()
            
#
# Update Q and policy
#
    print('main: sequenceHistory = ', sequenceHistory)
    print()
    print('main: newQ before update = ', newQ)
    for kState,val in sequenceHistory.iteritems():
        print('kState = ', kState)
        print('val = ', val)
        
        tmpQ = newQ[kState]
        
        for kAction, val2 in val.iteritems():
            tmpQ[kAction] = tmpQ[kAction] + 1.0/val2*(finalReward-tmpQ[kAction])
        newQ[kState]=tmpQ
        
    print()
    print('main: newQ after update = ', newQ)
    
#
# Update the policy
#    

    print()
    myEps=0.01
    print('main: policy before update = ', policy)
    for kState,val in newQ.iteritems():
        print()
        print('kState = ', kState)
        print('val = ', val)
        
        tmpPolicy = policy[kState]
        
        # find max Q result
        valActionMax = -float('inf')
        actionMax = ''
        for kQ, valQ in val.iteritems():
            print()
            print('main: kQ = ', kQ)
            print('main: valQ = ', valQ)
            print('main: valActionMax = ', valActionMax)
            if valQ > valActionMax:
                print('in here.')
                valActionMax=valQ
                actionMax=kQ
                print('valActionMax = ', valActionMax)
                print('actionMax = ', actionMax)
                
        print('main: final actionMax = ', actionMax)
       
        updatePolicy={}
        for pAction, pVal in tmpPolicy.iteritems():
            if pAction==actionMax:
                updatePolicy[pAction]=1.0-myEps
            else:
                updatePolicy[pAction]=myEps
        
        policy[kState]=updatePolicy
                     
    print()
    print('main: policy after update = ', policy)
    
    
###    flatCount=np.ndarray.flatten(gridStateCounter)
###    for iCount in range(len(flatCount)):
###        flatCount[iCount]=max(flatCount[iCount],1)
        
###    gridStateCounter=np.reshape(flatCount,gridStateCounter.shape)
    
###    print('main: gridStateCounter after update = ', gridStateCounter)
        
    # update values
###    gridWorld=np.divide(gridWorld,gridStateCounter)
###    print('main: gridWorld values after updates: ', gridWorld)

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