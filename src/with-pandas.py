import os
import sys
os.getcwd()
sys.path.append(os.getcwd() + '/src')

from NeuralNetwork import NeuralNetwork
import pandas as pd
import numpy as np

training_file = "/Users/amirmukeri/Downloads/mnist_train.csv"
test_file = "/Users/amirmukeri/Downloads/mnist_test.csv"


# training_file = "mnist_dataset/mnist_train_100.csv"
# test_file = "mnist_dataset/mnist_test_10.csv"
# training_file = "mnist_dataset/process_data_test.csv"
# test_file = "mnist_dataset/process_data_test.csv"


input_nodes = 784
hidden_nodes = 200
output_nodes = 10
learning_rate = 0.1

n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

training_data_file = pd.read_csv(training_file, header=None, index_col=None)


epochs = 5

for e in range(epochs):
    for i,row in training_data_file.iterrows():
        #all_values = record.split(',')
        #inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        row
        inputs = pd.Series(row[1:])
        inputs = inputs.apply(lambda x:(x/255.0 * 0.99) + 0.01)
        inputs
        targets = pd.Series(0.0,index=np.arange(10))
        targets
        targets[int(row[0])] = 0.99
        targets
        inputs.tolist()
        n.train(inputs.values.tolist(),targets.values.tolist())
        pass
    pass

test_data_file = pd.read_csv(test_file, header=None, index_col=None)
test_data_file.head()

scorecard = []

for i,row in test_data_file.iterrows():
    correct_label = int(row[0])
    correct_label
    inputs = pd.Series([row[1:]])
    inputs
    inputs = inputs.apply(lambda x: (x/255.0 * 0.99) + 0.01)
    inputs
    outputs = n.query(inputs.values.tolist())
    outputs
    label = np.argmax(outputs)
    label
    if(label == correct_label):
        scorecard.append(1)
    else:
        scorecard.append(0)
        pass
    pass

scorecard_array = np.asarray(scorecard)
print("Performance:", scorecard_array.sum()/scorecard_array.size)
