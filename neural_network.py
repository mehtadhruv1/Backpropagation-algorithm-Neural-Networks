#####################################################################################################################
#  Neural Network Backpropagation
# The neural network has two hidden layers
#  
#   Below are the meaning of symbols:
#   train - training dataset - can be a link to a URL or a local file
#         - you can assume the last column will the label column
#   train - test dataset - can be a link to a URL or a local file
#         - you can assume the last column will the label column
#   h1 - number of neurons in the first hidden layer
#   h2 - number of neurons in the second hidden layer
#   X - vector of features for each instance
#   y - output for each instance
#   w01, delta01, X01 - weights, updates and outputs for connection from layer 0 (input) to layer 1 (first hidden)
#   w12, delata12, X12 - weights, updates and outputs for connection from layer 1 (first hidden) to layer 2 (second hidden)
#   w23, delta23, X23 - weights, updates and outputs for connection from layer 2 (second hidden) to layer 3 (output layer)
#
#
#####################################################################################################################


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
from sklearn.model_selection import train_test_split


class NeuralNet:
    def __init__(self,Immunotherapy,activation_option, header = True, h1 = 4, h2 = 2):
      
        # train refers to the training dataset
        # test refers to the testing dataset
        # h1 and h2 represent the number of nodes in 1st and 2nd hidden layers

      
        
        dataset = pd.read_csv(Immunotherapy)
        
        ncols = len(dataset.columns)
        nrows = len(dataset.index)
    
        x = dataset.iloc[:, 0:(ncols -1)].values.reshape(nrows, ncols-1)
        y = dataset.iloc[:, (ncols-1)].values.reshape(nrows, 1)
        X_train, X_test, self.y_train, self.y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
       

        self.X_train, self.X_test = self.preprocess(X_train,X_test)

##        
        #
        # Find number of input and output layers from the dataset
        #
        input_layer_size = len(self.X_train[0])
        if not isinstance(self.y_train[0], np.ndarray):
            output_layer_size = 1
        else:
            output_layer_size = (self.y_train.shape[1])

        # assign random weights to matrices in network
        # number of weights connecting layers = (no. of nodes in previous layer) x (no. of nodes in following layer)
        self.w01 = 2 * np.random.random((input_layer_size, h1)) - 1

        self.X01 = self.X_train

        self.delta01 = np.zeros((input_layer_size, h1))

        self.w12 = 2 * np.random.random((h1, h2)) - 1

        self.X12 = np.zeros((len(self.X_train), h1))

        self.delta12 = np.zeros((h1, h2))

        self.w23 = 2 * np.random.random((h2, output_layer_size)) - 1

        self.X23 = np.zeros((len(self.X_train), h2))

        self.delta23 = np.zeros((h2, output_layer_size))

        self.deltaOut = np.zeros((output_layer_size, 1))
        #self.activation_option = input("Select which activation function to use for hidden and output layers: 1. Sigmoid  2. Tanh 3. Relu")
        self.activation_option= activation_option
        
   

    def __activation(self, x, activation):
        if activation == 1:
            self.__sigmoid(self, x)
        if activation == 2:
            self.__tanh(self, x)
        if activation == 3:
            self.__relu(self, x)


    

    def __activation_derivative(self, x, activation):
        if activation == 1:
            self.__sigmoid_derivative(self, x)
        if activation == 2:
            self.__tanh_derivative(self, x)
        if activation == 3:
            self.__relu_derivative(self, x)
# Sigmoid
    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # derivative of sigmoid function, indicates confidence about existing weight

    def __sigmoid_derivative(self, x):
        return x * (1 - x)
    
 # Tanh   
    def __tanh(self, x):
        return np.tanh(x)

    # derivative of sigmoid function, indicates confidence about existing weight

    def __tanh_derivative(self, x):
        return (1- np.power(x,2 ))
    
# Relu    
    def __relu(self, x):
       
        #print (x)
        x= np.maximum(0,x)
       
        
        
            
        return x
        

    # derivative of sigmoid function, indicates confidence about existing weight

    def __relu_derivative(self, x):
     
        x[x<=0] = 0
        x[x>0] = 1
        
        
        return x
        

  
    

    def preprocess(self,X_train,X_test):
        # scaling and normalization
        X_train[:,[0,4]]= X_train[:,[0,4]]
       
        X_train[:,[1,2,3,5,6]] = sc.fit_transform(X_train[:,[1,2,3,5,6]])
        X_test[:,[0,4]]= X_test[:,[0,4]]
       
        X_test[:,[1,2,3,5,6]] = sc.transform(X_test[:,[1,2,3,5,6]])
        
        # categorical attributes encoding can be done by using sklearn.preprocessing.OneHotEncoder. I have not implemented it as my selected dataset has already encoded the categorical attributes.
        #scaled_columns = sc.transform(scaled_columns)
        
#        print("Normalised training data ",X_train)
#        print("Normalised testing data ",X_test)
        #X=  np.concatenate([scaled_columns, encoded_columns], axis=1)

        return X_train,X_test

    # Below is the training function

    def train(self, learning_rate, max_iterations = 1000 ):
      
        for iteration in range(max_iterations):
            out = self.forward_pass(self.X_train,self.activation_option)

            error = 0.5 * np.power((out - self.y_train), 2)
            self.backward_pass(out,self.activation_option)
            update_layer2 = learning_rate * self.X23.T.dot(self.deltaOut)
            update_layer1 = learning_rate * self.X12.T.dot(self.delta23)
            update_input = learning_rate * self.X01.T.dot(self.delta12)

            self.w23 += update_layer2
            self.w12 += update_layer1
            self.w01 += update_input

        print("\n After " + str(max_iterations) + " iterations, the total error is " + str(np.sum(error)))
        print("\n The final weight vectors are (starting from input to output layers)\n ")
        print(self.w01)
        print()
        print(self.w12)
        print()
        print(self.w23)

    def forward_pass(self,X ,activation_option):
        # pass our inputs through our neural network
        
        in1 = np.dot(X, self.w01 )
        if int(activation_option) ==1:
            self.X12 = self.__sigmoid(in1)
        if int(activation_option) ==2:
            self.X12 = self.__tanh(in1)
        if int(activation_option) ==3:
            
            self.X12 = self.__relu(in1)
     
            
        in2 = np.dot(self.X12, self.w12)
        if int(activation_option) ==1:
            self.X23 = self.__sigmoid(in2)
        if int(activation_option)  ==2:
            self.X23 = self.__tanh(in2)
        if int(activation_option)  ==3:
            self.X23 = self.__relu(in2)
        
        in3 = np.dot(self.X23, self.w23)
        out =0
        if int(activation_option)  ==1:
            out = self.__sigmoid(in3)
        if int(activation_option)  ==2:
            out = self.__tanh(in3)
        if int(activation_option)  ==3:
            out = self.__relu(in3)
        return out



    def backward_pass(self, out, activation_option):
       
        self.compute_output_delta(out, activation_option)

        self.compute_hidden_layer2_delta(activation_option)

        self.compute_hidden_layer1_delta(activation_option)


    def compute_output_delta(self, out, activation):
        
        
        
        if int(activation) == 1:
            
            delta_output = (self.y_train - out) * (self.__sigmoid_derivative(out))
        
        
        elif int(activation) == 2:
            delta_output = (self.y_train - out) * (self.__tanh_derivative(out))
        else :
            
            delta_output = (self.y_train - out) * (self.__relu_derivative(out))
            

        self.deltaOut = delta_output



    def compute_hidden_layer2_delta(self, activation):
        
        
        
        if int(activation) == 1:
            delta_hidden_layer2 = (self.deltaOut.dot(self.w23.T)) * (self.__sigmoid_derivative(self.X23))
        elif int(activation) == 2:
            delta_hidden_layer2 = (self.deltaOut.dot(self.w23.T)) * (self.__tanh_derivative(self.X23))

        else:
            delta_hidden_layer2 = (self.deltaOut.dot(self.w23.T)) * (self.__relu_derivative(self.X23))

        self.delta23 = delta_hidden_layer2
       


    def compute_hidden_layer1_delta(self, activation="sigmoid"):
        
        if int(activation) == 1:
            delta_hidden_layer1 = (self.delta23.dot(self.w12.T)) * (self.__sigmoid_derivative(self.X12))
        elif int(activation) == 2:
            delta_hidden_layer1 = (self.delta23.dot(self.w12.T)) * (self.__tanh_derivative(self.X12))
        else:
            delta_hidden_layer1 = (self.delta23.dot(self.w12.T)) * (self.__relu_derivative(self.X12))
        self.delta12 = delta_hidden_layer1


#predict function for applying the trained model on the  test dataset.


    def predict(self,header = True):
        
        print("\n Training done, Testing time ")
   
        out= self.forward_pass(self.X_test,self.activation_option )
        print ("\n The error on test set is ",( str(np.sum(0.5 * np.power((out - self.y_test), 2)))))
        return 0


if __name__ == "__main__":
    
    print("\n Neural network error using sigmoid activation function \n ")
    print ("\n Learning rate = 0.05 \n ")
    neural_network = NeuralNet("Immunotherapy1.csv", 1)
    neural_network.train(0.05)
    neural_network.predict()
    
# =============================================================================
#     print ("\n Learning rate = 0.01")
#     neural_network = NeuralNet("Immunotherapy1.csv", 1)
#     neural_network.train(0.01)
#     neural_network.predict()
#     
#     print ("\n Learning rate = 0.1")
#     neural_network = NeuralNet("Immunotherapy1.csv", 1)
#     neural_network.train(0.1)
#     neural_network.predict()
#     
#     print ("\n Learning rate = 0.5")
#     neural_network = NeuralNet("Immunotherapy1.csv", 1)
#     neural_network.train(0.5)
#     neural_network.predict()
#     
#     print("\n Neural network error using tanh activation function \n ")
#     print ("\n Learning rate = 0.05 \n ")
#     neural_network = NeuralNet("Immunotherapy1.csv", 2)
#     neural_network.train(0.05)
#     neural_network.predict()
#     
#     print ("\n Learning rate = 0.01 \n")
#     neural_network = NeuralNet("Immunotherapy1.csv", 2)
#     neural_network.train(0.01)
#     neural_network.predict()
#     
#     print ("\n Learning rate = 0.1 \n")
#     neural_network = NeuralNet("Immunotherapy1.csv", 2)
#     neural_network.train(0.1)
#     neural_network.predict()
#     
#     print ("\n Learning rate = 0.5 \n")
#     neural_network = NeuralNet("Immunotherapy1.csv", 2)
#     neural_network.train(0.5)
#     neural_network.predict()
#     
#     print("\n Neural network error using relu activation function \n ")
#     print ("\n Learning rate = 0.05 \n ")
#     neural_network = NeuralNet("Immunotherapy1.csv", 3)
#     neural_network.train(0.05)
#     neural_network.predict()
#     
#     print ("\n Learning rate = 0.01\n")
#     neural_network = NeuralNet("Immunotherapy1.csv", 3)
#     neural_network.train(0.01)
#     neural_network.predict()
#     
#     print ("\n Learning rate = 0.1\n")
#     neural_network = NeuralNet("Immunotherapy1.csv", 3)
#     neural_network.train(0.1)
#     neural_network.predict()
#     
#     print ("\n Learning rate = 0.5\n")
#     neural_network = NeuralNet("Immunotherapy1.csv", 3)
#     neural_network.train(0.5)
#     neural_network.predict()
#     
# =============================================================================
    
    

