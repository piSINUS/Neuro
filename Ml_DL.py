import numpy as np
import scipy as scy
import pandas as pd
import matplotlib.pyplot as plt


class neuralNetwork:
    
    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate
        self.wih = (np.random.normal(0.0,pow(self.hnodes,-0.5)))
        self.who = (np.random.normal(0.0,pow(self.onodes,-0.5)))
        self.activation_function = lambda x:scy.special.expit(x) 

    def train(self,inputs_list,targets_list):
        inputs = np.array(inputs_list,ndmin = 2).T
        targets = np.array(targets_list,ndmin=2).T

        hidden_inputs = np.dot(self.wih,inputs)
        hidden_outputs = self.activation_function(hidden_inputs)


        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        output_errors = targets - final_outputs
        hidden_errors = np.dot((self.who.T,output_errors))

        self.who += self.lr * np.dot((output_errors*final_outputs *(1.0 - final_outputs)),np.transpose(hidden_outputs))
        self.wih += self.lr * np.dot((hidden_errors*hidden_outputs *(1.0 - hidden_outputs)),np.transpose(inputs))

    def query(self,inputs_list):
        inputs = np.array(inputs_list,ndmin=2).T
        hidden_inputs = np.dot(self.wih,inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.who,hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs

input_nodes = 784
hidden_nodes = 100
output_nodes = 10

learning_rate = 0.3

n = neuralNetwork( input_nodes,hidden_nodes,output_nodes,learning_rate)

print(n.query([1.0,0.5,-1.5]))

training_data_file = open('mnist_test_10.csv','r')
training_data_list = training_data_file.readlines()
training_data_file.close()

for record in training_data_list:
    all_values = record.split(',')

    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01

    targets[int(all_values[0])]  = 0.99

    n.train(inputs,targets)