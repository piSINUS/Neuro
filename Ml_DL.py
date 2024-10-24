import numpy as np
import scipy as scy
import pandas as pd
import matplotlib.pyplot as plt

# определение класса нс
class neuralNetwork:
    
    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate):
        # задаем кол - во узлов в входном,скрытом и выходном слое
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        # Матрицы весовых коэфф-ов связейБ wih и who.
        # Весовые коэфф-ты связей между узлом i и узлом j
        # следующего слоя обозначены как  w_i_j
        self.wih = (np.random.normal(0.0,pow(self.inodes,-0.5),(self.hnodes,self.inodes)))
        self.who = (np.random.normal(0.0,pow(self.hnodes,-0.5),(self.onodes, self.hnodes)))
        # коэффициент обучения 
        self.lr = learningrate
        # использование сигмоиды в качестве функции активации
        self.activation_function = lambda x:scy.special.expit(x) 

        pass
    # тренеровка нс
    def train(self,inputs_list,targets_list):
        # преобразование списка входных значений
        # в двухмерный массив
        inputs = np.array(inputs_list,ndmin = 2).T
        targets = np.array(targets_list,ndmin=2).T
        #  рассчитать входящие сигналы для скрытого слоя
        hidden_inputs = np.dot(self.wih,inputs)
        #  рассчитать исходящие сигналы для скрытого слоя
        hidden_outputs = self.activation_function(hidden_inputs)

        #  рассчитать входящие сигналы для выходного слоя 
        final_inputs = np.dot(self.who, hidden_outputs)
        #  рассчитать исодящие сигналы для выходного слоя 
        final_outputs = self.activation_function(final_inputs)

        # ошибки входного слоя = (целевое значение - фактическое значение)
        output_errors = targets - final_outputs
        # ошибки скрытого слоя - это ошибки output_errors, распределенные пропорционально весовым коэф связей и рекомбинированные на скрытых узлах
        hidden_errors = np.dot(self.who.T,output_errors)
       

        #  обновить весовые коэф для связей между скрыт и входными слоями
        self.who += self.lr * np.dot((output_errors*final_outputs *(1.0 - final_outputs)),np.transpose(hidden_outputs))
        #  обновить весовые коэф для связей между входными и скрытым слоям
        self.wih += self.lr * np.dot((hidden_errors*hidden_outputs *(1.0 - hidden_outputs)),np.transpose(inputs))

        pass
    # Опрос  нс 
    def query(self,inputs_list):
        # преобразование списка входных значений
        # в двухмерный массив
        inputs = np.array(inputs_list,ndmin=2).T

        #  рассчитать входящие сигналы для скрытого слоя
        hidden_inputs = np.dot(self.wih,inputs)
        #  рассчитать исходящие сигналы для скрытого слоя
        hidden_outputs = self.activation_function(hidden_inputs)

        #  рассчитать входящие сигналы для выходного слоя
        final_inputs = np.dot(self.who,hidden_outputs)
         #  рассчитать исодящие сигналы для выходного слоя 
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs

# кол-во входных выходных скрытых узлов
input_nodes = 784
hidden_nodes = 100
output_nodes = 10

# коэф обучения
learning_rate = 0.3

# Экземпляр нс 
n = neuralNetwork( input_nodes,hidden_nodes,output_nodes,learning_rate)

# Загрузить в список текстовый набор CSV файло
training_data_file = open('mnist_train_100.csv','r')
training_data_list = training_data_file.readlines()
training_data_file.close()

# перебрать все записи в трен наборе данных 
for record in training_data_list:
    # получить список значений, используя символы запятой в качестве разделителей
    all_values = record.split(',')

    # масштабировть и сместить входные значения
    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01

    # желаемого маркерного значения, равного 0,99
    targets = np.zeros(output_nodes) + 0.01
    
    #  all_values - целевое маркерное значение для данной записи
    targets[int(all_values[0])]  = 0.99

    n.train(inputs,targets)
    pass

a = n.query((np.asfarray(all_values[1:]) / 255.0 *0.99) + 0.01)
print(a)