import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

class NeuralNetwork(object):
    
    def __init__(self, n_in, n_layer1, n_layer2, n_out, learning_rate=0.01):
        '''__init__
        Class constructor: Initialize the parameters of the network including
        the learning rate, layer sizes, and each of the parameters
        of the model (weights, placeholders for activations, inputs, 
        deltas for gradients, and weight gradients). This method
        should also initialize the weights of your model randomly
            Input:
                n_in:          number of inputs
                n_layer1:      number of nodes in layer 1
                n_layer2:      number of nodes in layer 2
                n_out:         number of output nodes
                learning_rate: learning rate for gradient descent
            Output:
                none
        '''
        self.n_in = n_in
        self.n_layer1 = n_layer1
        self.n_layer2 = n_layer2
        self.n_out = n_out
        self.learning_rate = learning_rate

        # initialize weights
        self.W1 = np.random.randn(n_layer1, n_in)*0.01
        self.W2 = np.random.randn(n_layer2, n_layer1)*0.01
        self.W3 = np.random.randn(n_out, n_layer2)*0.01

        self.b1 = np.random.randn(n_layer1,1)*0.01
        self.b2 = np.random.randn(n_layer2,1)*0.01
        self.b3 = np.random.randn(n_out,1)*0.01
        # self.b1 = np.zeros((n_layer1,1))
        # self.b2 = np.zeros((n_layer2,1))
        # self.b3 = np.zeros((n_out,1))
        
    def forward_propagation(self, x):
        '''forward_propagation
        Takes a vector of your input data (one sample) and feeds
        it forward through the neural network, calculating activations and
        layer node values along the way.
            Input:
                x: a vector of data representing 1 sample [n_in x 1]
            Output:
                y_hat: a vector (or scaler of predictions) [n_out x 1]
                (typically n_out will be 1 for binary classification)
        '''
        # input layer
        self.z2 = np.dot(self.W1,x) + self.b1
        self.a2 = self.sigmoid(self.z2)

        # hidden layer 1
        self.z3 = np.dot(self.W2, self.a2)+self.b2
        self.a3 = self.sigmoid(self.z3)

        # hidden layer 2
        self.z4 = np.dot(self.W3, self.a3)+self.b3
        y_hat = self.sigmoid(self.z4)
        #print(self.z4, y_hat)

        return y_hat
    
    def compute_loss(self, X, y):
        '''compute_loss
        Computes the current loss/cost function of the neural network
        based on the weights and the data input into this function.
        To do so, it runs the X data through the network to generate
        predictions, then compares it to the target variable y using
        the cost/loss function
            Input:
                X: A matrix of N samples of data [N x n_in]
                y: Target variable [N x 1]
            Output:
                loss: a scalar measure of loss/cost
        '''
        # y_hat = []
        # for i in range(len(X)):
        #     x = X[i,:]
        #     y_hat.append(self.forward_propagation(x.reshape(-1,1)))
        # y_hat = np.array(y_hat)
        #x = X.reshape(-1, self.batch_size)
        y_hat = self.forward_propagation(X)
        loss = np.mean(-y * np.log(y_hat) - (1 - y) * np.log(1 - y_hat))
        return loss
    
    def backpropagate(self, x, y):
        '''backpropagate
        Backpropagate the error from one sample determining the gradients
        with respect to each of the weights in the network. The steps for
        this algorithm are:
            1. Run a forward pass of the model to get the activations 
               Corresponding to x and get the loss functionof the model 
               predictions compared to the target variable y
            2. Compute the deltas (see lecture notes) and values of the
               gradient with respect to each weight in each layer moving
               backwards through the network
    
            Input:
                x: A vector of 1 samples of data [n_in x 1]
                y: Target variable [scalar]
            Output:
                loss: a scalar measure of th loss/cost associated with x,y
                      and the current model weights
        '''
        
        # Forward propagation
#         self.x = x.reshape(-1, 1)
#         self.z1 = self.W1.dot(np.vstack([self.x, 1]))
#         self.a1 = self.sigmoid(self.z1)
#         self.z2 = self.W2.dot(np.vstack([self.a1, 1]))
#         self.a2 = self.sigmoid(self.z2)
#         self.z3 = self.W3.dot(np.vstack([self.a2, 1]))
#         self.y_hat = self.sigmoid(self.z3)
        #x = x.reshape(-1, self.batch_size)
        #print(x.shape)
        y_hat = self.forward_propagation(x)
        #dz4 = (y_hat-y)*self.sigmoid_derivative(self.z4)
        # 批量样本直接是向量的计算
        dz4 = y_hat-y
        # 样本求和体现在矩阵里，体现在单个样本的向量的外积形成的矩阵
        # 多个样本则为多个这样的矩阵求和
        # 在实现时，也可以看成多个样本形成的矩阵和矩阵的乘积（矩阵乘积运算的一种解释即为对应的列向量和行向量的外积之和）
        dW3 = np.dot(dz4,self.a3.T)/self.batch_size
        # 对于偏置梯度 db，则直接对误差矩阵 dZ 跨样本维度求平均即可
        db3 = np.mean(dz4,axis=1,keepdims=True)
        #print(dz4.shape, db3.shape, dW3.shape)
        
        dz3 = np.dot(self.W3.T,dz4)*self.sigmoid_derivative(self.z3)
        dW2 = np.dot(dz3,self.a2.T)/self.batch_size
        #db2 = dz3
        db2 = np.mean(dz3,axis=1,keepdims=True)
     
        dz2 = np.dot(self.W2.T,dz3)*self.sigmoid_derivative(self.z2)
        dW1 = np.dot(dz2,x.T)/self.batch_size
        #db1 = dz2
        db1 = np.mean(dz2,axis=1,keepdims=True)
      
        self.W3 -= self.learning_rate * dW3
        self.W2 -= self.learning_rate * dW2
        self.W1 -= self.learning_rate * dW1
  
        self.b3 -= self.learning_rate * db3
        self.b2 -= self.learning_rate * db2
        self.b1 -= self.learning_rate * db1

        loss = -y * np.log(y_hat) - (1 - y) * np.log(1 - y_hat)
        return np.mean(loss)
    
    def stochastic_gradient_descent_step(self):
        '''stochastic_gradient_descent_step [OPTIONAL - you may also do this
        directly in backpropagate]
        Using the gradient values computed by backpropagate, update each
        weight value of the model according to the familiar stochastic
        gradient descent update equation.
        
        Input: none
        Output: none
        '''
        #idx = np.random.randint(0, len(self.X_train))
        #Y = np.einsum('kij, lk -> lij', X, A)
        idx_start = self.cur_epoch_iter*self.batch_size
        x = self.X_train[self.idx[idx_start:idx_start+self.batch_size]].T
        y = self.y_train[self.idx[idx_start:idx_start+self.batch_size]].reshape(1,-1)
        #print(x,y)
        # Compute gradients and update weights using backpropagation
        self.backpropagate(x, y)
        # Get a random data point from the training set
        
        # for idx in range(len(self.X_train)//16):
        #     #idx = np.random.randint(0, len(self.X_train))
        #     #Y = np.einsum('kij, lk -> lij', X, A)
        #     x = self.X_train[idx*16:(idx+1)*16]
        #     y = self.y_train[idx*16:(idx+1)*16]
        
        #     # Compute gradients and update weights using backpropagation
        #     self.backpropagate(x, y)
    
    def fit(self, X, y, X_val, y_val, batch_size = 16, max_epochs=500, learning_rate=0.01, get_validation_loss=False):
        '''fit
            Input:
                X: A matrix of N samples of data [N x n_in]
                y: Target variable [N x 1]
            Output:
                training_loss:   Vector of training loss values at the end of each epoch
                validation_loss: Vector of validation loss values at the end of each epoch
                                 [optional output if get_validation_loss==True]
        '''
        #self.X_train = X
        #self.y_train = y
        self.learning_rate = learning_rate
        self.batch_size = batch_size
    
        training_loss = []
        validation_loss = []
    
        # Split the data into training and validation sets
        #self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X, y, test_size=0.2)
        self.X_train, self.X_val, self.y_train, self.y_val = X,X_val,y,y_val
        self.idx = [i for i in range(len(self.X_train))]
        for i in range(max_epochs):
            np.random.shuffle(self.idx)
            for j in range(len(self.X_train)//self.batch_size):
                self.cur_epoch_iter = j
                self.stochastic_gradient_descent_step()
            # Compute training loss for this epoch
            train_loss = self.compute_loss(self.X_train.T, self.y_train.reshape(1,-1))
            print('train_loss', train_loss)
            training_loss.append(train_loss)
        
            # Compute validation loss for this epoch if desired
            if get_validation_loss:
                val_loss = self.compute_loss(self.X_val.T,self.y_val.reshape(1,-1))
                print('val_loss', val_loss)
                validation_loss.append(val_loss)
    
        #print(self.W1,self.W2,self.W3,self.b1,self.b2,self.b3)
        if get_validation_loss:
            return training_loss, validation_loss
        else:
            return training_loss
            
    def predict_proba(self, X):
        '''predict_proba
        Compute the output of the neural network for each sample in X, with the last layer's
        sigmoid activation providing an estimate of the target output between 0 and 1
            Input:
                X: A matrix of N samples of data [N x n_in]
            Output:
                y_hat: A vector of class predictions between 0 and 1 [N x 1]
        '''
        return self.forward_propagation(X)
        # y_hat = []
        # for x in X:
        #     y_pred = self.forward_propagation(x)[0][0]
        #     y_hat.append(y_pred)
        # return np.array(y_hat)
    
    def predict(self, X, decision_thresh=0.5):
        '''predict
        Compute the output of the neural network prediction for 
        each sample in X, with the last layer's sigmoid activation 
        providing an estimate of the target output between 0 and 1, 
        then thresholding that prediction based on decision_thresh
        to produce a binary class prediction
            Input:
                X: A matrix of N samples of data [N x n_in]
                decision_threshold: threshold for the class confidence score
                                    of predict_proba for binarizing the output
            Output:
                y_hat: A vector of class predictions of either 0 or 1 [N x 1]
        '''
    
        y_hat = self.predict_proba(X)
        y_hat_binary = np.where(y_hat > decision_thresh, 1, 0).T
        return y_hat_binary
    
    def sigmoid(self, X):
        '''sigmoid
        Compute the sigmoid function for each value in matrix X
            Input:
                X: A matrix of any size [m x n]
            Output:
                X_sigmoid: A matrix [m x n] where each entry corresponds to the
                           entry of X after applying the sigmoid function
        '''
        return 1.0 / (1.0 + np.exp(-X))
    
    def sigmoid_derivative(self, X):
        '''sigmoid_derivative
        Compute the sigmoid derivative function for each value in matrix X
            Input:
                X: A matrix of any size [m x n]
            Output:
                X_sigmoid: A matrix [m x n] where each entry corresponds to the
                           entry of X after applying the sigmoid derivative function
        '''
        return self.sigmoid(X) * (1 - self.sigmoid(X))


def scaler_fit(X):
    global mean,scale
    mean=X.mean(axis=0)
    scale=X.std(axis=0)
    scale[scale<np.finfo(scale.dtype).eps]=1.0
def scaler_transform(X):
    return (X-mean)/scale

# Set random seed for reproducibility
np.random.seed(42)
# Create training, validation, and test datasets
N_train = 1000
N_test = 200
# N_train = 500
# N_test = 100
# X, y = make_moons(N_train + N_test, noise=0.20, random_state=42)

X, y = make_moons(N_train + N_test, noise=0.20, random_state=42)
#X, y = make_blobs(N_train + N_test, centers=2, n_features=2, random_state=3)
# scaler = StandardScaler()
# X = scaler.fit_transform(X)
scaler_fit(X)
X=scaler_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=N_test, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


plt.scatter(X_train[:,0], X_train[:, 1], c=y_train, edgecolors='white', marker='s')
plt.show()


myNN = NeuralNetwork(n_in=2, n_layer1=10, n_layer2=10, n_out=1, learning_rate=0.01)
# Initialize the MLPClassifier

# Train the model and collect the cost values for each epoch
training_loss, validation_loss = myNN.fit(X_train, y_train, X_val, y_val, batch_size=16, max_epochs=3000, learning_rate=0.1, get_validation_loss=True)
# Plot the learning curves

plt.plot(training_loss, label='Training cost')
plt.plot(validation_loss, label='Validation cost')
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.title('Cost function plot')
plt.legend()
plt.show()


y_hat = myNN.predict(X_train.T)
y_hat_test = myNN.predict(X_test.T)
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))

# Plot the decision boundary on the training data
scatter0 = axs[0][0].scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.Spectral)
#axs[0].set_title('Decision boundary for training data')
#xx, yy = np.meshgrid(np.linspace(-2, 2, 100), np.linspace(-2, 2, 100))
#Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
#contour0 = axs[0].contourf(xx, yy, Z, levels=[0, 0.5, 1], alpha=0.5, cmap=plt.cm.Spectral)
#axs[0].contour(xx, yy, Z, levels=[0.5], colors='k')
#axs[0].set_xticks([])
#axs[0].set_yticks([])

#Plot the decision boundary on the validation data
scatter1 = axs[0][1].scatter(X_train[:, 0], X_train[:, 1], c=y_hat, cmap=plt.cm.Spectral)
#axs[1].set_title('Decision boundary for validation data')
#contour1 = axs[1].contourf(xx, yy, Z, levels=[0, 0.5, 1], alpha=0.5, cmap=plt.cm.Spectral)
#axs[1].contour(xx, yy, Z, levels=[0.5], colors='k')
#axs[1].set_xticks([])
#axs[1].set_yticks([])
scatter2 = axs[1][0].scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.Spectral)
scatter3 = axs[1][1].scatter(X_test[:, 0], X_test[:, 1], c=y_hat_test, cmap=plt.cm.Spectral)
#plt.scatter(X_train[:,0], X_train[:, 1], c=y_train, edgecolors='white', marker='s')
plt.show()