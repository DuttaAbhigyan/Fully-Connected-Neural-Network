#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 11:38:07 2018

@author: abhigyan
"""

import numpy as np


"""Layer class: Used by the Neural_Network class to create new layers for the Neural Network"""

class layer(object):

#Initializer Methods
    
    def __init__(self, nodes, activation, alpha = None):
        self.activation = activation
        self.nodes = nodes
        self.alpha = alpha
                
    def initializeVariables(self, previousLayerNodes):
        self.weights = np.random.normal(size = (previousLayerNodes, self.nodes), loc = 0.0, scale = 1.0)
        self.bias = np.random.normal(size = (1, self.nodes), loc = 0.0, scale = 1.0)

#End


#Getter Methods        
    
    def getActivation(self):
        return self.activation
        
    def getWeightsAndBias(self):
        return self.weights, self.bias

#End


#Forward Propagation        
        
    def applyActivation(self, inputs):
        self.inputs = inputs
        self.z = np.dot(inputs, self.weights) + self.bias
        self.inputShape = inputs.shape
        
        if(self.activation == 'sigmoid'):
            self.a = np.power((1 + np.exp(-self.z)), -1)
            
        elif(self.activation == 'relu'):
            self.a = np.where(self.z > 0, self.z, 0)
            
        elif(self.activation == 'leakyrelu'):
            self.a = np.where(self.z > 0, self.z, self.alpha * self.z) 
            
        return self.a

#End

#Backpropagation Through Neural Network, Updation of weights and  calculation of Downstream gradients            
    
    def backpropagation(self, gradientWRTz):
        self.gradientWRTz = gradientWRTz
        self.summedGradientWRTz = np.sum(gradientWRTz, axis = 0, keepdims = True)
        self.weightGradient = np.dot(self.inputs.T, self.gradientWRTz) / self.inputShape[0]
        self.biasGradient = self.summedGradientWRTz / self.inputShape[0]    
                
    def updateWeights(self, learningRate):
        self.weights = self.weights - learningRate * self.weightGradient
        self.bias = self.bias - self.biasGradient
                
    def calculateGradientWRTpreviousZ(self, activation = None):    
        self.gradientWRTw = np.dot(self.gradientWRTz, self.weights.T)
        
        if(activation == 'sigmoid'):
            self.gradientWRTpreviousZ = self.gradientWRTw * self.inputs * (1 - self.inputs)
            
        elif(activation == 'relu'):
            self.gradientWRTpreviousZ = self.gradientWRTw * np.where(self.inputs > 0, 1, 0)
            
        elif(activation == 'laekyrelu'):
            self.gradientWRTpreviousZ = self.gradientWRTw * np.where(self.inputs > 0, 1, self.alpha)
            
        elif(activation == None):
            return self.gradientWRTw
            
        return self.gradientWRTpreviousZ
    
#End    
    

"""Final Layer class: Used by the Neural_Network class to create final layer for the Neural Network"""
    
class finalLayer(object):

#Initializer Methods
    
    def __init__(self, classes, outputFunction, errorFunction, alpha = None):
        self.classes = classes
        self.outputFunction = outputFunction
        self.errorFunction = errorFunction
        self.alpha = alpha
           
    def initializeVariables(self, previousLayerNodes):
        self.weights = np.random.normal(size = (previousLayerNodes, self.classes), loc = 0.0, scale = 1.0)
        self.bias = np.random.normal(size = (1, self.classes), loc = 0.0, scale = 1.0)
        
#End
        

#Getter Methods
        
    def getActivation(self):
        return self.outputFunction
    
    
    def getWeightsAndBias(self):
        return self.weights, self.bias

#End
        
    
#Forward Propagation        
      
    def applyOutputFunction(self, inputs):
        self.inputs = inputs
        self.z = np.dot(self.inputs, self.weights) + self.bias
        self.inputShape = inputs.shape
        
        if(self.outputFunction == 'sigmoid'):
            self.a = np.power((1 + np.exp(-self.z)), -1)
            
        elif(self.outputFunction == 'relu'):
            self.a = np.where(self.z > 0, self.z, 0)
            
        elif(self.outputFunction == 'leakyrelu'):
            self.a = np.where(self.z > 0, self.z, self.alpha * self.z)
            
        elif(self.outputFunction == 'softmax'):
            self.a = np.exp(self.z) / np.sum(np.exp(self.z))
            
        return self.a

#End
        

#Calculation of Loss, Backpropagation through Neural Net, Weight Updation, Calculation of Downstream Gradients
    
    def calculateLoss(self, target):
        self.loss = 0
        self.target = target

        if(self.errorFunction == 'cross_entropy'):
            
            self.loss = -np.sum(self.target * np.log(self.a) + (1 - self.target) * np.log(1 - self.a)) / self.inputShape[0]
            
        if(self.errorFunction == 'squared_error'):
            self.loss = np.sum((self.a - self.target) ** 2) / self.inputShape[0]
        
        return self.loss
    
    
    def backpropagation(self):
        self.error = self.a - self.target
        
        if(not(self.outputFunction == 'sigmoid' or self.outputFunction == 'softmax')):
            print("Loss is only available for sigmoid and softmax activation functions")
        
        elif(self.errorFunction == 'cross_entropy'):
            self.gradientWRTz = np.copy(self.error)
            self.summedGradientWRTz = np.sum(self.error, axis = 0, keepdims = True)
            self.weightGradient = np.dot(self.inputs.T, self.gradientWRTz) / self.inputShape[0]
            self.biasGradient = self.summedGradientWRTz / self.inputShape[0]
            
        elif(self.errorFunction == 'squared_error'):
            self.gradientWRTz = 2 * (self.a - self.target) * self.a * (1 - self.a)
            self.summedGradientWRTz = np.sum(self.error, axis = 0, keepdims = True)
            self.weightGradient = (self.inputs.T * self.summedGradientWRTz) / self.inputShape[0]
            self.biasGradient = self.summedGradientWRTz / self.inputShape[0]
        
    
    def updateWeights(self, learningRate):
        self.weights = self.weights - learningRate * self.weightGradient
        self.bias = self.bias - self.biasGradient
        
    
    def calculateGradientWRTpreviousZ(self, activation):
        self.gradientWRTpreviousZ = None
        self.gradientWRTw = np.dot(self.gradientWRTz, self.weights.T)
        
        if(activation == 'sigmoid'):
            self.gradientWRTpreviousZ = self.gradientWRTw * self.inputs * (1 - self.inputs)
            
        elif(activation == 'relu'):
            self.gradientWRTpreviousZ = self.gradientWRTw * np.where(self.inputs > 0, 1, 0)
            
        elif(activation == 'leakyrelu'):
            self.gradientWRTpreviousZ = self.gradientWRTw * np.where(self.inputs > 0, 1, self.alpha)
            
        elif(activation == None):
            return self.gradientWRTw
            
        return self.gradientWRTpreviousZ
    
#End    
        
        


"""Main Controller class which creates the entire Neural Network based on the parameters given by the user"""

class Neural_Network(layer, finalLayer):
    
    def __init__(self, layers, nodes, activations, errorFunction, alpha = None, optimizer = 'GradientDescent', learningRate = 0.1 ):
        self.optimizer = optimizer
        self.layers = layers
        self.nodes = nodes
        self.learningRate = learningRate
        self.NNlayers = []
        self.NNlayerOutputs = []
        self.errorFunction = errorFunction
        self.alpha = alpha
        
        if((layers != len(nodes)) or (layers != len(activations))):
            print("Invalid Neural Network Parameters")
            
        else:
            for i in range(0, layers):
                if(i == layers-1):
                    l = finalLayer(nodes[i], activations[i], self.errorFunction, self.alpha)
                    self.NNlayers.append(l)
                    break
                l = layer(nodes[i], activations[i], alpha)
                self.NNlayers.append(l)
    

    def initializeNN(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets
        self.inputShape = inputs.shape
        
        for j in range(0, self.layers):
            if(j == 0):
                self.NNlayers[j].initializeVariables(self.inputShape[1])
                output = self.NNlayers[j].applyActivation(self.inputs)
                self.NNlayerOutputs.append(output)
            elif(j == self.layers - 1):
                self.NNlayers[j].initializeVariables(self.nodes[j-1])
                output = self.NNlayers[j].applyOutputFunction(self.NNlayerOutputs[j-1])
                self.NNlayerOutputs.append(output)
            else:                
                self.NNlayers[j].initializeVariables(self.nodes[j-1])
                output = self.NNlayers[j].applyActivation(self.NNlayerOutputs[j-1])
                self.NNlayerOutputs.append(output)
        
        
    def run_Neural_Network(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets
        self.inputShape = inputs.shape
        
        for j in range(0, self.layers):
            if(j == 0):
                self.NNlayers[j].initializeVariables(self.inputShape[1])
                output = self.NNlayers[j].applyActivation(self.inputs)
                self.NNlayerOutputs.append(output)
            elif(j == self.layers - 1):
                self.NNlayers[j].initializeVariables(self.nodes[j-1])
                output = self.NNlayers[j].applyOutputFunction(self.NNlayerOutputs[j-1])
                self.NNlayerOutputs.append(output)
            else:                
                self.NNlayers[j].initializeVariables(self.nodes[j-1])
                output = self.NNlayers[j].applyActivation(self.NNlayerOutputs[j-1])
                self.NNlayerOutputs.append(output)
         
        self.loss = self.NNlayers[-1].calculateLoss(self.targets)
        self.gradientWRTz = []
        for j in range(self.layers-1, -1, -1):
            if(j == self.layers-1):
                self.NNlayers[j].backpropagation()
                self.gradientWRTz.append(self.NNlayers[j].calculateGradientWRTpreviousZ(self.NNlayers[j-1].getActivation()))
            elif(j == 0):
                self.NNlayers[j].backpropagation(self.gradientWRTz[self.layers-j-2])
            else:
                self.NNlayers[j].backpropagation(self.gradientWRTz[self.layers-j-2])
                self.gradientWRTz.append(self.NNlayers[j].calculateGradientWRTpreviousZ(self.NNlayers[j-1].getActivation()))
                
                
    def gradient_descent_optimizer(self):
        for i in range(0, self.layers):
            self.NNlayers[i].updateWeights(self.learningRate)
                
#End
            

        
        
      
    
