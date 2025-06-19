import NN
import numpy as np 


#data is read from the datafile and stored in a python list
data_file_train = open("mnist_train.csv", 'r')
data_list_train = data_file_train.readlines()
data_file_train.close()


#opens the training dataset
data_file_test = open("mnist_test.csv", 'r')
data_list_test = data_file_test.readlines()
data_file_test.close()


input_nodes = 784
hidden_nodes = 150
output_nodes = 10
learning_rate = 0.3

#instance of neural network with the parameters defined above is created
nn = NN.NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

#preparation for mini-batch gradient descent:

batch_size = 32 
num_epochs = 5

x = []
y = []
#list formatting loop that loops through every element in the dataset and formats it to fit in the input-array(x) or the target-array(y)
for record in data_list_train:
    values = record.split(',')
    #inputs are rescaled to fit into the "comfort zone" of the neural network, which is between 0.01 and 1.00
    scaled_inputs = (np.asfarray(values[1:])/ 255.0 * 0.99) + 0.01
    #target array is created with the floor value of 0.01 except for the correct label which should be 0.99
    targets = np.zeros(output_nodes) + 0.01
    targets[int(values[0])] = 0.99


    #scaled inputs are appended to the input-array(x)
    x.append(scaled_inputs)
    #target vectors are appended to the target-array(y)
    y.append(targets)

#data-lists are converted into numpy arrays for easier slicing
x = np.array(x)
y = np.array(y)

n = len(x)
#order of indices in the x and y arrays is shuffled randomly for each training-epoch to prevent the network from learning the order of elements
for epoch in range(num_epochs):
    print(epoch)
    #index-array containing every index in range(0, 59999)
    indices = np.arange(n)
    #index-array is shuffled, which assures which assures randomness when creating batches
    np.random.shuffle(indices)

    #order of the elements in the x and y array is changed according to the shuffled-indices-list
    #the order of matching pairs among the elements in the x and y array however remains
    x_shuffled = x[indices]
    y_shuffled = y[indices]
    #training loop that iterates over the data with step-size being the batch-size to select a "batch-size amount" of elements through list slicing from the current index up to the next index
    for i in range(0, n, batch_size):
        x_batch = x_shuffled[i:i+batch_size]
        y_batch = y_shuffled[i:i+batch_size]
        nn.train(x_batch, y_batch)


scorecard = []
#querys the network and calculates accuracy 
for record in data_list_test:
    values = record.split(',')

    correct_label = int(values[0])
    scaled_inputs = (np.asfarray(values[1:])/ 255.0 * 0.99) + 0.01
    outputs = nn.query(scaled_inputs)
    
    #max value of the networks outputs is the label the network guessed
    label = np.argmax(outputs)
    
    if label == correct_label:
        scorecard.append(1)
    else: 
        scorecard.append(0)

accuracy = sum(scorecard) / len(data_list_test)
print("Accuracy:", accuracy)