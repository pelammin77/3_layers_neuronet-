"""
file: Neural_network.py
Author: Petri Lamminaho
email: lammpe77@gmail.com
"""
from numpy import random, array, exp, dot

class Neural_network():

    def __init__(self):
        """
        constructor
        """
        random.seed(1)  # antaa ainna samat numerot kun ohjelma käy
        num_peers_2_layer = 5
        num_peers_3_layer = 4

        self.synaptic_weights1 = 2 * random.random((3, num_peers_2_layer)) - 1  # create neutron first layer
        self.synaptic_weights2 = 2 * random.random((num_peers_2_layer, num_peers_3_layer)) - 1
        self.synaptic_weights3 = 2 * random.random((num_peers_3_layer, 1)) - 1
#-----------------------------------------------------------------------------------------------------------
    def __sigmoid(self, x):
        """
        private function
        sigmoid function pass the data
         and normalize data to 1 or 0
        :param x:
        :return: 1 or 0
        """
        return 1 / (1 + exp(-x))
#--------------------------------------------------------------------------------------------------
    def __sigmoid_derivative(self, x):
        """
        private function
        :param x:
        :return derivative function :
        """
        return x * (1 - x)
#----------------------------------------------------------------------------------
    def train(self, training_inputs, training_outputs, num_of_training_iterations):
        for iteration in range(num_of_training_iterations):
            # pass training set through our neural network
            # a2 means the activations fed to second layer
            a2 = self.__sigmoid(dot(training_inputs, self.synaptic_weights1))
            a3 = self.__sigmoid(dot(a2, self.synaptic_weights2))
            output = self.__sigmoid(dot(a3, self.synaptic_weights3))

            # calculate 'error'
            d4 = (training_outputs - output) * self.__sigmoid_derivative(output)


            d3 = dot(self.synaptic_weights3, d4.T) * (self.__sigmoid_derivative(a3).T)
            d2 = dot(self.synaptic_weights2, d3) * (self.__sigmoid_derivative(a2).T)

            # get adjustments (gradients) for each layer
            adjustment3 = dot(a3.T, d4)
            adjustment2 = dot(a2.T, d3.T)
            adjustment1 = dot(training_inputs.T, d2.T)

            # adjust weights accordingly
            self.synaptic_weights1 += adjustment1
            self.synaptic_weights2 += adjustment2
            self.synaptic_weights3 += adjustment3




  #---------------------------------------------------------------------------------------------
    def think(self, inputs):
        """

        :param inputs:
        :return: output
        take outputs
        pass inputs to next layer
        returns output
         """
        a2 = self.__sigmoid(dot(inputs, self.synaptic_weights1))
        a3 = self.__sigmoid(dot(a2, self.synaptic_weights2))
        output = self.__sigmoid(dot(a3, self.synaptic_weights3))
        return output


#-------------------------------------------------------------------------------------------------------
"""
main function 
"""

if __name__ == "__main__":
    nn = Neural_network()
    print("random start weights")
    print(nn.synaptic_weights1)
    training_data_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    training_data_outputs = array([[0, 1, 1, 0]]).T
    print("Training data inputs:")
    print(training_data_inputs)
    print("Training data outputs:")
    print(training_data_outputs)
    nn.train(training_data_inputs, training_data_outputs, 10000)
    print("New weights after training: ")
    print(nn.synaptic_weights1)

    # Test the neural network with a new input
    print ("Trying new input data [1, 0, 0 ] -> ?: ( output should be close 1")
    print("result:",nn.think(array([1, 0, 0]))) #output: 0.99650838
