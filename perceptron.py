import numpy as np

def sigmoid (x):
	return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
	return x*(1-x)

EPOCHS = 5000
# learning_rate = 0.001

#standard gated inputs
input_layer  =   np.array([[0,0,1],
						[0,1,1],
						[1,0,1],
						[1,1,1]])

labels = np.array([[0,0,1,1]]).T

#fixes a single value for random weights
np.random.seed(420)
weights = np.random.rand(3,1)
# training_baises  = np.random.rand(1)

print ("input weights:")
print (weights ,'\n')


#training
for epochs in range(EPOCHS):

	output_values = sigmoid(np.dot(input_layer ,weights))

	error = labels - output_values
	adjustments = error*sigmoid_derivative(output_values)
	weights += np.dot(input_layer.T , adjustments)

print(output_values , "\n")
print(weights, "\n")