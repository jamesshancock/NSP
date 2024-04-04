'''
This file contains the visualization functions for the nurse scheduling problem
'''
import numpy as np
import matplotlib.pyplot as plt


def solutionToGrid(solution,parameters):
    '''
    This returns a grid of the solution
    '''
    grid = np.zeros((parameters['N'],parameters['D']))
    for i in range(parameters['N']*parameters['D']):
        grid[i//parameters['D'],i%parameters['D']] = solution[i]
    return grid

def gridPlotter(solution , plot, parameters):
    '''
    This plots the grid of the solution
    '''
    grid = solutionToGrid(solution,parameters)    
    if plot == True:

        plt.imshow(grid, cmap='Blues')
        plt.yticks([])
        plt.xticks([])
        xoffset = -0.4 + grid.shape[0]

        for i in range(grid.shape[1]):
            plt.text(i, xoffset, 'Shift {}'.format(i+1), ha='center', va='top', color='black')

        for i in range(grid.shape[0]):
            plt.text(-0.6, i, 'Nurse {}'.format(i+1), ha='right', va='center', color='black')
        
        plt.title("Schedule for "+str(grid.shape[0])+" nurses, with "+str(grid.shape[1])+" shifts.") 

        plt.show()
        plt.yticks([])
        plt.xticks([])
        xoffset = -0.4 + grid.shape[0]

        for i in range(grid.shape[1]):
            plt.text(i, xoffset, 'Shift {}'.format(i+1), ha='center', va='top', color='black')

        for i in range(grid.shape[0]):
            plt.text(-0.6, i, 'Nurse {}'.format(i+1), ha='right', va='center', color='black')
    
        plt.title("Schedule for "+str(grid.shape[0])+" nurses, with "+str(grid.shape[1])+" shifts.") 

        plt.show()
    return None