import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("Data/train.csv")
df = np.array(df)
r, c = df.shape
np.random.shuffle(df)

data = df[0:1000].T
X_data = data[1:c]
X = X_data / 255
y = data[0]

data_train = df[1000:r].T
y_train = data_train[0]
X_Train = data_train[1:c]
X_Train = X_Train / 255
_,r_train = X_Train.shape


def init_params():
    w1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    w2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return w1, b1, w2, b2

def ReLU(z):
    return np.maximum(z, 0)

def softmax(z):
    A = np.exp(z) / sum(np.exp(z))
    return A

def forward_prop(w1, b1, w2, b2, x):
    z1 = w1.dot(x) + b1
    A1 = ReLU(z1)
    z2 = w2.dot(A1) + b2
    A2 = softmax(z2)
    return z1, A1, z2, A2

def ReLU_d(z):
    return z > 0

def one_hot_enc(y):
    one_hot_enc_y = np.zeros((y.size, y.max() + 1))
    one_hot_enc_y[np.arange(y.size), y] = 1
    one_hot_enc_y = one_hot_enc_y.T
    return one_hot_enc_y

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot_enc(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / r * dZ2.dot(A1.T)
    db2 = 1 / r * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_d(Z1)
    dW1 = 1 / r * dZ1.dot(X.T)
    db1 = 1 / r * np.sum(dZ1)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2    
    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))
    return W1, b1, W2, b2

W1, b1, W2, b2 = gradient_descent(X_Train, y_train, 0.10, 500)


def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def test_prediction(index, W1, b1, W2, b2):
    current_image = X_Train[:, index, None]
    prediction = make_predictions(X_Train[:, index, None], W1, b1, W2, b2)
    label = y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()
    
test_prediction(0, W1, b1, W2, b2)