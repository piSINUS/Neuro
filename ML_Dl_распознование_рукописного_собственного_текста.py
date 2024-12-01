import numpy as np
import scipy as scy
import pandas as pd
import matplotlib.pyplot as plt
import scipy.misc
import glob
import imageio.v3


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
training_data_file = open('minst_datas/mnist_train_100.csv','r')
training_data_list = training_data_file.readlines()
training_data_file.close()

epochs =  10 
for e in range(epochs):

    # перебрать все записи в трен наборе данных 
    for record in training_data_list:
        # получить список значений, используя символы запятой в качестве разделителей
        all_values = record.split(',')

        # масштабировть и сместить входные значения
        inputs = (np.asarray(all_values[1:]) / 255.0 * 0.99) + 0.01

        # желаемого маркерного значения, равного 0,99
        targets = np.zeros(output_nodes) + 0.01
        
        #  all_values - целевое маркерное значение для данной записи
        targets[int(all_values[0])]  = 0.99

        n.train(inputs,targets)
        pass
    pass

our_own_dataset =[]


for image_file_name in glob.glob('data/0.png'):
    
    
    label = int(image_file_name[-5:-4])
    
    
    print ("loading ... ", image_file_name)
    img_array = imageio.v3.imread(image_file_name, mode='F')
    
   
    img_data  = 255.0 - img_array.reshape(784)
    
    
    img_data = (img_data / 255.0 * 0.99) + 0.01
    print(np.min(img_data))
    print(np.max(img_data))
    
    
    record = np.append(label,img_data)
    our_own_dataset.append(record)
    
    pass

item = 0


plt.pyplot.imshow(our_own_dataset[item][1:].reshape(28,28), cmap='Greys', interpolation='None')

correct_label = our_own_dataset[item][0]

inputs = our_own_dataset[item][1:]


outputs = n.query(inputs)
print (outputs)


label = np.argmax(outputs)
print("network says ", label)

if (label == correct_label):
    print ("match!")
else:
    print ("no match!")
    pass

# img_array = scipy.misc.imread()
# test_data_file = open("mnist_test_10.csv",'r')
# test_data_list  = test_data_file.readlines()
# test_data_file.close()
# # тестирование нс
# # журнал оценки работы сети, первоначально пустой
# scorecard = []

# #перебрать все записи в тестовом наборе данных 
# for record in test_data_list:
#     # получить список значений, используя символы запятой в качестве разделителей
#     all_values = record.split(',')
#     # правильгный ответ - ервое значение
#     correct_label = int(all_values[0])
#     print(correct_label,'Истинный маркер')
#     # масштабировать и сместить входные значения
#     inputs=(np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
#     # опрос сети
#     outputs = n.query(inputs)
#     # индекс наибольшего значения является маркерным значением
#     label = np.argmax(outputs)
#     print(label,'ответ сети')
#     # присоединить оценку ответа сети к концу списка
#     if (label == correct_label):
#         # в случае правельного ответа сети присоединить к списку значение 1
#         scorecard.append(1)
#     else:
#         # в случае правельного ответа сети присоединить к списку значение 0
#         scorecard.append(0)
#         pass
#     pass

# scorecard_array = np.asarray(scorecard)

# print(f"эффективность = {scorecard_array.sum() / scorecard_array.size}")
# # a = n.query((np.asfarray(all_values[1:]) / 255.0 *0.99) + 0.01)

# if __name__ == "__main__":
#     print("This script is in the top-level code environment")