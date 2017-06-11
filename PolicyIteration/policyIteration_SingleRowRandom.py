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
# Create grid world
#
gridRows=1
gridCols=7
termLocations = []
stepReward=0  # cost to move one grid location

gridWorld = np.zeros([gridRows,gridCols])
#
# Initialize values
#
termLocations.append(np.array([0,0]))  # set first grid element to be terminal node
termLocations.append(np.array([0,6]))  # set last grid element to be terminal node
#for iRow in range(gridWorld.shape[0]):
#    for jCol in range(gridWorld.shape[1]):
#        gridWorld[iRow,jCol]=iRow*gridWorld.shape[1]+jCol
print('initial gridworld:')
print(gridWorld)
print()

#
# Create value lookup function
#
def valLookup(grid,direction,refPosition):
    
    if direction=='u': # up
        testPosition=refPosition+np.array([-1,0]) # up one row
        if testPosition[0]<0:
            testPosition=refPosition  # prevent moving off world boundary
    
    elif direction=='d': #down
        testPosition=refPosition+np.array([1,0]) # up one row
        if testPosition[0]>=grid.shape[0]:
            testPosition=refPosition  # prevent moving off world boundary

    elif direction=='l': #left
        testPosition=refPosition+np.array([0,-1]) # up one row
        if testPosition[1]<0:
            testPosition=refPosition  # prevent moving off world boundary

    elif direction=='r': # right
        testPosition=refPosition+np.array([0,1]) # up one row
        if testPosition[1]>=grid.shape[1]:
            testPosition=refPosition  # prevent moving off world boundary

    else: # error
        sys.exit('Please enter a valid direction.') 
    
    # normal exit
###    print('valLookup: refPosition', refPosition)
###    print('valLookup: testPosition', testPosition)
    return grid[testPosition[0],testPosition[1]]

#
# Loop over states and update values according to default policy
#
numLoops=3
for iLoop in range(numLoops):

    testDirSet = {'l','r'}
    for iRow in range(gridWorld.shape[0]):
        for iCol in range(gridWorld.shape[1]):
            refLocation = np.array([iRow,iCol])
            maxReward=-float('inf')
            maxArgDir=''
            inTerm=False
            # special handling of terminal location
            for iLoc in termLocations:
                if np.array_equal(np.array(iLoc), refLocation):
                    inTerm=True
                    print('main: refLocation=', refLocation)
                    print('main: We are in a terminal location.')
        
            # Compute rewards and values
            if inTerm:
                maxReward=0 
                newFinalVal = gridWorld[refLocation[0],refLocation[1]] # same value
            else:
                for testDir in testDirSet: # check all directions for max reward
                    newVal = valLookup(gridWorld, testDir, refLocation)
                    #Special case of right transition to end state gives +1 reward
                    if np.array_equal(refLocation,np.array([0,5])) and testDir=='r':
                        stepReward=1
                    currReward = stepReward + newVal
#                    if currReward > maxReward:
                    if True:
                        maxReward = currReward
                        maxArgDir = testDir
                
                
            print('main: refLocation = ', refLocation)
            print('main: maxArgDir = ', maxArgDir)
            print('main: maxReward = ', maxReward)
            print()
        
            gridWorld[iRow,iCol]=maxReward
            
#
# show results
#
    print('updated gridworld:')
    print(gridWorld)
    print()

#
# Update policy if desired
#

#
# Display results
#

#
# End of script
#