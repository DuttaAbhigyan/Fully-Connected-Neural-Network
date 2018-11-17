#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 21:45:11 2018

@author: abhigyan
"""
("""UNSTABLE VERSION""")

import numpy as np

#Default Parameters

"""Initialization --> Gaussian Random
    Learning Rate --> 0.1
    Optimiser     --> Gradient Descent"""

"Layer class: Used by the Neural_Network class to create new layers for the Neural Network"

class layer(object):

#Initializer Methods
    
    def __init__(self, nodes, activation, learningRate,
                 momentumCoeff = None, alpha = None):   #alpha is the slope of negative part in leaky ReLu 
        self.activation = activation                                    #or parameters which are required in other activations 
        self.nodes = nodes                                              #like prelu, elu and selu 
        self.mu = momentumCoeff                         
        self.alpha = alpha                         
                
    def initializeVariables(self, previousLayerNodes):
        self.weights = np.random.normal(size = (previousLayerNodes, self.nodes),  
                                        loc = 0.0, scale = 1.0)
        self.bias = np.random.normal(size = (1, self.nodes), loc = 0.0, scale = 1.0)
        self.weightGradientHistory = np.zeros((previousLayerNodes, self.nodes))
        self.biasGradientHistory = np.zeros((1, self.nodes))

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
            
        elif(self.activation == 'hsigmoid'):
            hSigmoidApprox = 0.2*self.z + 0.5
            self.a = np.clip(hSigmoidApprox, 0, 1)
            
        elif(self.activation == 'leakyrelu'):
            self.a = np.where(self.z > 0, self.z, self.alpha * self.z) 
        
        elif(self.activation == 'tanh'):
            self.a = (np.exp(self.z) - np.exp(-self.z)) * np.power(np.exp(+self.z) + np.exp(-self.z), -1)
        
        elif(self.activation == 'elu'):
            self.a = np.where(self.z > 0, self.z, self.alpha * (np.exp(self.z) - 1))
            
        elif(self.activation == 'prelu'):
            self.a = np.where(self.z > 0, self.z, self.alpha * self.z)
                        
        return self.a

#End
        
#Weight manipulations based on the optimiser used
        
    def maintainGradientHistory(self):
        self.weightGradientHistory = self.weightUpdate
        self.biasGradientHistory = self.biasUpdate
            
    def createWeightUpdates(self, learningRate, optimiser):
        if(optimiser == 'gradient_descent'):
            self.weightUpdate = learningRate * self.weightGradient
            self.biasUpdate = learningRate * self.biasGradient
            
        elif(optimiser == 'gradient_descent_momentum'):
            self.weightUpdate = self.mu * self.weightGradientHistory + (1-self.mu) * learningRate * self.weightGradient
            self.biasUpdate = self.mu * self.biasGradientHistory + (1-self.mu) * learningRate * self.biasGradient
            
        elif(optimiser == 'RMS_prop'):
            pass
#End

#Backpropagation Through Neural Network, Updation of weights and  calculation of Downstream gradients            
    
    def backpropagation(self, gradientWRTz):
        self.gradientWRTz = gradientWRTz
        self.summedGradientWRTz = np.sum(gradientWRTz, axis = 0, keepdims = True)
        self.weightGradient = np.dot(self.inputs.T, self.gradientWRTz) / self.inputShape[0]
        self.biasGradient = self.summedGradientWRTz / self.inputShape[0]    
                
    def updateWeights(self):
        self.weights = self.weights - self.weightUpdate
        self.bias = self.bias - self.biasUpdate
        
                
    def calculateGradientWRTpreviousZ(self, activation = None):    #Calculate gradient WRT 'z' of previous layer
        self.gradientWRTw = np.dot(self.gradientWRTz, self.weights.T)
        
        if(activation == 'sigmoid'):
            self.gradientWRTpreviousZ = self.gradientWRTw * self.inputs * (1 - self.inputs)
            
        elif(activation == 'relu'):
            self.gradientWRTpreviousZ = self.gradientWRTw * np.where(self.inputs > 0, 1, 0)
            
        elif(activation == 'hsigmoid'):
            self.gradientWRTpreviousZ = self.gradientWRTw * np.where(np.abs(self.inputs - 0.5) < 0.5, 0.2, 0)
            
        elif(activation == 'leakyrelu'):
            self.gradientWRTpreviousZ = self.gradientWRTw * np.where(self.inputs > 0, 1, self.alpha)
            
        elif(activation == 'tanh'):
            self.gradientWRTpreviousZ = self.gradientWRTw * (1 - np.power(self.inputs, 2))
            
        elif(activation == 'elu'):
            self.gradientWRTpreviousZ = self.gradientWRTw * np.where(self.inputs > 0, 1, self.inputs + self.alpha)
            
        elif(activation == None):
            return self.gradientWRTw
            
        return self.gradientWRTpreviousZ
    
#End    
    

"Final Layer class: Used by the Neural_Network class to create final layer for the Neural Network"
    
class finalLayer(object):

#Initializer Methods
    
    def __init__(self, classes, outputFunction, errorFunction, 
                 momentumCoeff = None, alpha = None):
        self.classes = classes
        self.outputFunction = outputFunction
        self.errorFunction = errorFunction
        self.mu = momentumCoeff
        self.alpha = alpha
           
    def initializeVariables(self, previousLayerNodes):
        self.weights = np.random.normal(size = (previousLayerNodes, self.classes), 
                                        loc = 0.0, scale = 1.0)
        self.bias = np.random.normal(size = (1, self.classes), loc = 0.0, scale = 1.0)
        self.weightGradientHistory = np.zeros((previousLayerNodes, self.classes))
        self.biasGradientHistory = np.zeros((1, self.classes))
#End
        

#Getter Methods
        
    def getActivation(self):
        return self.outputFunction
        
    def getWeightsAndBias(self):
        return self.weights, self.bias
    
    def getResult(self):
        return self.a

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
            self.a = np.exp(self.z) / np.sum(np.exp(self.z), axis = 1, keepdims = True)
            
        return self.a

#End

        
#Weight manipulations based on the optimiser used
        
    def maintainGradientHistory(self):
        self.weightGradientHistory = self.weightUpdate
        self.biasGradientHistory = self.biasUpdate
            
    def createWeightUpdates(self, learningRate, optimiser):
        if(optimiser == 'gradient_descent'):
            self.weightUpdate = learningRate * self.weightGradient
            self.biasUpdate = learningRate * self.biasGradient
            
        elif(optimiser == 'gradient_descent_momentum'):
            self.weightUpdate = self.mu * self.weightGradientHistory + (1-self.mu) * learningRate * self.weightGradient
            self.biasUpdate = self.mu * self.biasGradientHistory + (1-self.mu) * learningRate * self.biasGradient
            
        elif(optimiser == 'RMS_prop'):
            pass
#End
        

#Calculation of Loss, Backpropagation through Neural Net, Weight Updation, Calculation of Downstream Gradients
    
    def calculateLoss(self, targets):
        self.loss = 0
        self.targets = targets

        if(self.errorFunction == 'cross_entropy' and self.outputFunction == 'sigmoid'):
            self.loss = -np.sum(self.targets * np.log(self.a) + (1 - self.targets) * np.log(1 - self.a)) / self.inputShape[0]
            
        elif(self.errorFunction == 'cross_entropy' and self.outputFunction == 'softmax'):
            self.loss = -np.sum(self.targets * np.log(self.a)) / self.inputShape[0]
            
        elif(self.errorFunction == 'squared_error'):
            self.loss = np.sum((self.a - self.targets) ** 2) / self.inputShape[0]
        
        return self.loss
    
    
    def backpropagation(self):
        self.error = self.a - self.targets
        
        if(not(self.outputFunction == 'sigmoid' or self.outputFunction == 'softmax')):
            print("Loss is only available for sigmoid and softmax activation functions")
        
        elif(self.errorFunction == 'cross_entropy'):
            self.gradientWRTz = np.copy(self.error)
            self.summedGradientWRTz = np.sum(self.error, axis = 0, keepdims = True)
            self.weightGradient = np.dot(self.inputs.T, self.gradientWRTz) / self.inputShape[0]
            self.biasGradient = self.summedGradientWRTz / self.inputShape[0]
            
        elif(self.errorFunction == 'squared_error'):
            self.gradientWRTz = 2 * (self.a - self.targets) * self.a * (1 - self.a)
            self.summedGradientWRTz = np.sum(self.error, axis = 0, keepdims = True)
            self.weightGradient = (self.inputs.T * self.summedGradientWRTz) / self.inputShape[0]
            self.biasGradient = self.summedGradientWRTz / self.inputShape[0]
        
    
    def updateWeights(self):
        self.weights = self.weights - self.weightUpdate
        self.bias = self.bias - self.biasUpdate
        
    
    def calculateGradientWRTpreviousZ(self, activation):
        self.gradientWRTpreviousZ = None
        self.gradientWRTw = np.dot(self.gradientWRTz, self.weights.T)
        
        if(activation == 'sigmoid'):
            self.gradientWRTpreviousZ = self.gradientWRTw * self.inputs * (1 - self.inputs)
            
        if(activation == 'hsigmoid'):
            self.gradientWRTpreviousZ = self.gradientWRTw * np.where(np.abs(self.inputs - 0.5) < 0.5, 0.2, 0)
            
        elif(activation == 'relu'):
            self.gradientWRTpreviousZ = self.gradientWRTw * np.where(self.inputs > 0, 1, 0)
            
        elif(activation == 'leakyrelu'):
            self.gradientWRTpreviousZ = self.gradientWRTw * np.where(self.inputs > 0, 1, self.alpha)
            
        elif(activation == 'tanh'):
            self.gradientWRTpreviousZ = self.gradientWRTw * (1 - np.power(self.inputs, 2))
            
        elif(activation == 'elu'):
            self.gradientWRTpreviousZ = self.gradientWRTw * np.where(self.inputs > 0, 1, self.inputs + self.alpha)
            
        elif(activation == None):
            return self.gradientWRTw
            
        return self.gradientWRTpreviousZ
    
#End    
        
        


"Main creator class which creates the entire Neural Network based on the parameters given by the user"

class Neural_Network(layer, finalLayer):
    
    def __init__(self, layers, nodes, activations, errorFunction, alpha = 0.01, 
                 optimizer = 'GradientDescent', momentumCoeff = 0.9, learningRate = 0.1, lrDecay = None, decayRate = 0.0 ):
        self.optimizer = optimizer
        self.layers = layers
        self.nodes = nodes
        self.learningRate = learningRate
        self.NNlayers = []
        self.NNlayerOutputs = []
        self.errorFunction = errorFunction
        self.mu = momentumCoeff
        self.alpha = alpha
        self.lrDecay = lrDecay
        self.decayRate = decayRate
        
        if((layers != len(nodes)) or (layers != len(activations))):
            print("Invalid Neural Network Parameters")
            
        else:
            for i in range(0, layers):
                if(i == layers-1):
                    l = finalLayer(nodes[i], activations[i], self.errorFunction, self.mu, self.alpha)
                    self.NNlayers.append(l)
                    break
                l = layer(nodes[i], activations[i], self.mu, self.alpha)
                self.NNlayers.append(l)
    

#Neural Network Inititializer function
                
    def initializeNN(self, inputs, targets, epochs):
        self.inputs = inputs
        self.targets = targets
        self.inputShape = inputs.shape
        self.epochs = epochs
        
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
                        
        

#Function which will run the Neural Network
                
    def run_Neural_Network(self):        
        
        for i in range(0, self.epochs):
            self.gradientWRTz = []
            self.NNlayerOutputs = []
            
        #Forward Propagation
                
            for j in range(0, self.layers):
                if(j == 0):
                    output = self.NNlayers[j].applyActivation(self.inputs)
                    self.NNlayerOutputs.append(output)
                elif(j == self.layers - 1):
                    output = self.NNlayers[j].applyOutputFunction(self.NNlayerOutputs[j-1])
                    self.NNlayerOutputs.append(output)
                else:                
                    output = self.NNlayers[j].applyActivation(self.NNlayerOutputs[j-1])
                    self.NNlayerOutputs.append(output)
        
            #Loss Calculation
            
            self.loss = self.NNlayers[-1].calculateLoss(self.targets)
            print(self.loss)
            self.accuracy_calculator()
            self.F_Score_calculator()
        
            #Backpropagation
            
            for j in range(self.layers-1, -1, -1):
                if(j == self.layers-1):
                    self.NNlayers[j].backpropagation()
                    self.gradientWRTz.append(self.NNlayers[j].calculateGradientWRTpreviousZ(self.NNlayers[j-1].getActivation()))
                elif(j == 0):
                    self.NNlayers[j].backpropagation(self.gradientWRTz[self.layers-j-2])
                else:
                    self.NNlayers[j].backpropagation(self.gradientWRTz[self.layers-j-2])
                    self.gradientWRTz.append(self.NNlayers[j].calculateGradientWRTpreviousZ(self.NNlayers[j-1].getActivation()))
            
            #Learning Rate Decay
            
            if(self.lrDecay == None):
                self.currentLearningRate = self.learningRate
            elif(self.lrDecay == 'first_order_time'):
                self.currentLearningRate = self.learningRate/(1+self.decayRate*i)
            elif(self.lrDecay == 'second_order_time'):   
                self.currentLearningRate = self.learningRate/(1+self.decayRate*(i**2))
            elif(self.lrDecay == 'exponential_decay'):
                self.currentLearningRate = self.learningRate*np.exp(-self.decayRate*i)
            
            #Optimization
            
            if(self.optimizer == 'GradientDescent'):
                self.gradient_descent_optimizer()
            elif(self.optimizer == 'GradientDescentWithMomentum'):
                self.momentum_descent_optimizer()
                

    #Optimiser functions
                
    def gradient_descent_optimizer(self):
        for i in range(0, self.layers):
            self.NNlayers[i].createWeightUpdates(self.currentLearningRate, 'gradient_descent')
            self.NNlayers[i].updateWeights()
            
    def momentum_descent_optimizer(self):
        for i in range(0, self.layers):
            self.NNlayers[i].createWeightUpdates(self.currentLearningRate, 'gradient_descent_momentum')
            self.NNlayers[i].updateWeights()
            self.NNlayers[i].maintainGradientHistory()
            
            
    #Score functions accuracy and F_score
    
    def accuracy_calculator(self):
        self.hypothesis = self.NNlayers[-1].getResult()
        
        if(self.NNlayers[-1].getActivation() == 'sigmoid' and self.nodes[-1] == 1):
            self.accuracyMatrix = np.round(self.hypothesis) == self.targets
            self.accuracyMatrix = self.accuracyMatrix.astype(dtype = np.int32)
            self.accuracy = np.sum(self.accuracyMatrix) / len(self.targets)
            print(self.accuracy)
        else:
            self.accuracyMatrix = np.argmax(self.hypothesis, axis = 1)  == np.argmax(self.targets, axis = 1)
            self.accuracyMatrix.astype(dtype = np.int32)
            self.accuracy = np.sum(self.accuracyMatrix) / len(self.targets)
            print(self.accuracy)
            
            
    def calculatePrecision(self, predictions, target):
        TP = np.sum(predictions & target)
        FP = np.sum(predictions & np.abs(target - 1))
        return TP/(TP+FP)
    
    
    def calculateRecall(self, predictions, target):
        TP = np.sum(predictions & target)
        FN = np.sum(np.abs(predictions - 1) & target)
        return TP/(TP+FN)
    
        
    def F_Score_calculator(self, averaging ='macro'):       #For more info check out https://sebastianraschka.com/faq/docs/multiclass-metric.html
        self.hypothesis = self.NNlayers[-1].getResult()
        predictions = np.array(np.round(self.hypothesis), dtype = np.int16)
        self.targets = np.array(self.targets, dtype = np.int16)
        
        if(self.NNlayers[-1].getActivation() == 'sigmoid' and self.nodes[-1] == 1):
            precision = self.calculatePrecision(predictions, self.targets) 
            recall = self.calculateRecall(predictions, self.targets) 
            self.F_score = (precision * recall)/(precision + recall)
        else:
            precision = np.array([])
            recall = np.array([])
            for i in range(self.targets.shape[1]):
                precision = np.append(precision, self.calculatePrecision(predictions[:, i], self.targets[:, i]))
                recall = np.append(recall, self.calculateRecall(predictions[:, i], self.targets[:, i]))
                
            if(averaging == 'macro'):
                averagePrecision = np.average(precision)
                averageRecall = np.average(recall)
                
            
            self.F_score = (averagePrecision * averageRecall)/(averagePrecision + averageRecall)
        print('F_score: ' + str(self.F_score))
                 
        
#End
            
    
