import numpy as np
import scipy as scy

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
        inputs = np.arrau(inputs_list,ndmin = 2).T
        targets = np.array(targets_list,ndmin=2).T

        hidden_inputs = np.dot(self.wih,inputs)
        hidden_outputs = self.activation_function(hidden_inputs)


        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)


    def query(self,inputs_list):
        inputs = np.array(inputs_list,ndmin=2).T
        hidden_inputs = np.dot(self.wih,inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.who,hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs

input_nodes = 3
hidden_nodes =3
output_nodes = 3

learning_rate = 0.3

n = neuralNetwork( input_nodes,hidden_nodes,output_nodes,learning_rate)

print(n.query([1.0,0.5,-1.5]))
