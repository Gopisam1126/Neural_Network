## Neural Network from Scratch

This repository presents a fully NumPy-based implementation of a feed-forward neural network for handwritten digit classification, **without** any deep-learning frameworks like TensorFlow or PyTorch.

## Features

- **End-to-end training** on the MNIST-style dataset via mini-batch gradient descent  
- **Custom layers** implemented from scratch:  
  - Dense (fully-connected)  
  - ReLU activation and its gradient  
  - Softmax output with cross-entropy loss  
  - Batch Normalization (inference/training modes)  
  - Dropout regularization  
- **Optimizers**:  
  - Vanilla SGD  
  - SGD with momentum  
- **Weight initialization** via Xavier (Glorot) scheme  
- **Visualization**: Preview individual predictions as 28×28 grayscale images using Matplotlib  

##Dataset

[Digit Recognizer](https://www.kaggle.com/c/digit-recognizer/data) dataset from Kaggle, containing 42,000 labeled 28×28 grayscale images of handwritten digits 0–9.

```bash
pip install pandas numpy matplotlib
```
```bash
kaggle competitions download -c digit-recognizer -p Data
unzip Data/digit-recognizer.zip -d Data
```
