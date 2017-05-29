import os

location_data = '/home/project/dynamic_neural_network/Pokemon.csv'
inference_column = 'Type 1'
input_columns = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
current_location = os.getcwd()

learning_rate = 0.01

num_layers = 5
# size of list = size num_layers
size_layers = [2048, 512, 128, 64, 16]

enable_dropout = False
# between (0, 1]
# size of list = size num_layers
dropout_probability = [0.1, 0.1, 0.1, 0.1, 0.1]

# only support {tanh, relu, sigmoid}
# size of list = size num_layers
activation_functions = ['relu', 'relu', 'relu', 'sigmoid', 'sigmoid']

# only support {tanh, relu, sigmoid, softmax}
last_activation_function = 'softmax'
# only support {MSE, cross_entropy}
loss_function = 'cross_entropy'

# only support {gradientdescent, adagrad, adam, rmsprop}
optimizer = 'adam'

enable_penalty = False
penalty = 0.0005

batch_size = 10
epoch = 1000
checkpoint = 50
split_test = 0.2