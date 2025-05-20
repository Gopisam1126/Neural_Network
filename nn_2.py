import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

batch_size = 64
epochs = 20
lr = 0.01
momentum = 0.9
drop_prob = 0.2
l2_lambda = 1e-4

def one_hot(y, num_classes=10):
    Y = np.zeros((num_classes, y.size))
    Y[y, np.arange(y.size)] = 1
    return Y

def softmax(z):
    ez = np.exp(z - np.max(z, axis=0, keepdims=True))
    return ez / ez.sum(axis=0, keepdims=True)

def relu(z):    
    return np.maximum(0, z)

def drelu(z):   
    return (z > 0).astype(float)

def xavier_init(size_in, size_out):
    bound = np.sqrt(6/(size_in + size_out))
    return np.random.uniform(-bound, bound, (size_out, size_in))

class BatchNorm:
    def __init__(self, dim, momentum=0.9, eps=1e-5):
        self.gamma = np.ones((dim,1))
        self.beta  = np.zeros((dim,1))
        self.mom, self.eps = momentum, eps
        self.running_mean = np.zeros((dim,1))
        self.running_var  = np.ones((dim,1))
        
    def forward(self, X, train=True):
        if train:
            mu = X.mean(axis=1, keepdims=True)
            var= X.var(axis=1, keepdims=True)
            
            self.norm = (X - mu) / np.sqrt(var + self.eps)
            out = self.gamma * self.norm + self.beta
            
            self.running_mean = self.mom*self.running_mean + (1-self.mom)*mu
            self.running_var  = self.mom*self.running_var  + (1-self.mom)*var
        else:
            norm = (X - self.running_mean) / np.sqrt(self.running_var + self.eps)
            out = self.gamma * norm + self.beta
        return out
    def backward(self, d_out):
        
        N = d_out.shape[1]
        dbeta  = d_out.sum(axis=1, keepdims=True)
        dgamma = (self.norm * d_out).sum(axis=1, keepdims=True)
        dnorm  = self.gamma * d_out
        var = self.running_var
        dX = (1/N) * (1/np.sqrt(var + self.eps)) * (N*dnorm 
            - dnorm.sum(axis=1,keepdims=True)
            - self.norm*(dnorm * self.norm).sum(axis=1,keepdims=True)
        )
        return dX, dgamma, dbeta

df = pd.read_csv("Data/train.csv").values
np.random.shuffle(df)

X = df[:,1:]/255.0
y = df[:,0].astype(int)

split = int(0.8 * len(y))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

X_train, y_train = X_train.T, y_train
X_test,  y_test  = X_test.T,  y_test

W1 = xavier_init(784,128)
b1 = np.zeros((128,1))
bn1= BatchNorm(128)

W2 = xavier_init(128,64)
b2 = np.zeros((64,1))
bn2= BatchNorm(64)

W3 = xavier_init(64,10)
b3 = np.zeros((10,1))

vW1 = np.zeros_like(W1); vb1 = np.zeros_like(b1)
vW2 = np.zeros_like(W2); vb2 = np.zeros_like(b2)
vW3 = np.zeros_like(W3); vb3 = np.zeros_like(b3)

# loop
for epoch in range(epochs):
    perm = np.random.permutation(X_train.shape[1])
    Xs, ys = X_train[:,perm], y_train[perm]
    for i in range(0, Xs.shape[1], batch_size):
        xb = Xs[:,i:i+batch_size]
        yb = ys[i:i+batch_size]
        Yb = one_hot(yb)

        # forwrad pass
        z1 = W1.dot(xb) + b1
        a1 = bn1.forward(z1, train=True)
        r1 = relu(a1)
        m1 = (np.random.rand(*r1.shape) > drop_prob) / (1-drop_prob)
        d1 = r1 * m1

        z2 = W2.dot(d1) + b2
        a2 = bn2.forward(z2, train=True)
        r2 = relu(a2)
        m2 = (np.random.rand(*r2.shape) > drop_prob) / (1-drop_prob)
        d2 = r2 * m2

        z3 = W3.dot(d2) + b3
        out= softmax(z3)

        # backward pass
        dz3  = out - Yb
        dW3 = dz3.dot(d2.T)/batch_size + l2_lambda*W3
        db3 = dz3.sum(axis=1,keepdims=True)/batch_size

        dd2 = W3.T.dot(dz3)
        da2 = dd2 * m2
        dbn2, dgamma2, dbeta2 = bn2.backward(da2)
        dz2 = drelu(a2) * dbn2

        dW2 = dz2.dot(d1.T)/batch_size + l2_lambda*W2
        db2 = dz2.sum(axis=1,keepdims=True)/batch_size

        dd1 = W2.T.dot(dz2)
        da1 = dd1 * m1
        dbn1, dgamma1, dbeta1 = bn1.backward(da1)
        dz1 = drelu(a1) * dbn1

        dW1 = dz1.dot(xb.T)/batch_size + l2_lambda*W1
        db1 = dz1.sum(axis=1,keepdims=True)/batch_size

        vW3 = momentum*vW3 + lr*dW3; W3 -= vW3
        vb3 = momentum*vb3 + lr*db3; b3 -= vb3
        vW2 = momentum*vW2 + lr*dW2; W2 -= vW2
        vb2 = momentum*vb2 + lr*db2; b2 -= vb2
        vW1 = momentum*vW1 + lr*dW1; W1 -= vW1
        vb1 = momentum*vb1 + lr*db1; b1 -= vb1

    # 1
    z1_test = W1.dot(X_test) + b1
    bn1_out = bn1.forward(z1_test, train=False)
    r1_test = relu(bn1_out)
    # 2
    z2_test = W2.dot(r1_test) + b2
    bn2_out = bn2.forward(z2_test, train=False)
    r2_test = relu(bn2_out)
    # Output layer
    z3_test = W3.dot(r2_test) + b3
    preds_test = np.argmax(softmax(z3_test), axis=0)

    acc = np.mean(preds_test == y_test)
    print(f"Epoch {epoch+1}/{epochs} — Test Accuracy: {acc:.4f}")

def test_prediction(index):
    x = X_test[:, index:index+1]
    
    z1 = W1.dot(x) + b1
    a1 = bn1.forward(z1, train=False)
    r1 = relu(a1)
    z2 = W2.dot(r1) + b2
    a2 = bn2.forward(z2, train=False)
    r2 = relu(a2)
    z3 = W3.dot(r2) + b3
    pred = np.argmax(softmax(z3), axis=0)[0]
    true = y_test[index]
    print(f"Prediction: {pred}, True Label: {true}")

    img = (x * 255).reshape(28,28)
    plt.imshow(img, cmap='gray', interpolation='nearest')
    plt.axis('off')
    plt.show()

test_prediction(0)
