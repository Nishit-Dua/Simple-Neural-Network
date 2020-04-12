import numpy as np

class NeuralNetwork():
	def __init__(self):		
		np.random.seed(420)
		self.weights = np.random.rand(3,1)
		# print(self.weights)
		
	def sigmoid (self,x):
		return 1 / (1 + np.exp(-x))

	def sigmoid_derivative(self,x):
		return x*(1-x)

	def train(self, input_layer, labels, EPOCHS = 5000):

		for epochs in range(EPOCHS):

			output_values = self.sigmoid(np.dot(input_layer ,self.weights))

			error = labels - output_values
			adjustments = np.dot(input_layer.T ,error*self.sigmoid_derivative(output_values))

			self.weights += adjustments

	def predict(self, inputs) :
		# inputs = inputs.astype(float)
		self.inputs = inputs
		output = self.sigmoid(np.dot(self.inputs, self.weights))
		return output


if __name__ == '__main__':

	neural_network = NeuralNetwork()

	input_layer  =   np.array([[0,0,1],
							[0,1,1],
							[1,0,1],
							[1,1,1]])

	labels = np.array([[0,0,1,1]]).T

	print("random weights :\n {}".format(neural_network.weights))

	neural_network.train(input_layer, labels)

	A = float(input ("\nenter input A : "))
	B = float(input ("enter input B : "))
	C = float(input ("enter input C : "))

	print("\nInputs to predict upon : ", A, B, C , "\n")

	output = neural_network.predict([A,B,C])
	print("Weights After Training:\n{}".format(neural_network.weights), "\n")
	print("After training : {}".format(output))