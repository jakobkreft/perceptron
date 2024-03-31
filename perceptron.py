import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class Layer:
    def __init__(self, n_inputs, n_outputs):
        # ustvarimo naključne uteži in odmike
        self.weights = np.random.randn(n_inputs, n_outputs) * 0.01
        self.biases = np.zeros((1, n_outputs))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)

class SigmoidActivation:
    #sigmoidna aktivacijska funkcija
    def forward(self, inputs):
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))

    def backward(self, dvalues):
        self.dinputs = dvalues * (1 - self.output) * self.output


class ReLUActivation:
    #Rectified linear: lompljena linearna
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

class SoftmaxActivation:
    # Softmax aktivacijska funkcija za zadnji sloj.
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)

    def backward(self, dvalues, y_true):
        self.dinputs = dvalues.copy()
        self.dinputs[range(len(dvalues)), y_true.argmax(axis=1)] -= 1
        self.dinputs = self.dinputs / len(dvalues)

class CategoricalCrossentropyLoss:
    #kategorična križna entropija izgub
    def compute_loss(self, y_pred, y_true):
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        correct_confidences = y_pred_clipped[range(len(y_pred)), y_true.argmax(axis=1)]
        return -np.log(correct_confidences)

    def compute_mean_loss(self, y_pred, y_true):
        return np.mean(self.compute_loss(y_pred, y_true))

class MomentumOptimizer:
    #razred za izračun momenta.
    def __init__(self, learning_rate=0.001, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.momentums = {}

    def update_params(self, layer, layer_index):
        if layer_index not in self.momentums:
            self.momentums[layer_index] = [np.zeros_like(layer.weights), np.zeros_like(layer.biases)]

        # popravimo uteži 
        self.momentums[layer_index][0] = self.momentum * self.momentums[layer_index][0] - self.learning_rate * layer.dweights
        layer.weights += self.momentums[layer_index][0]

        # popravimo odmike
        self.momentums[layer_index][1] = self.momentum * self.momentums[layer_index][1] - self.learning_rate * layer.dbiases
        layer.biases += self.momentums[layer_index][1]

class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.optimizer = None

    def add(self, layer):
        self.layers.append(layer)

    def set(self, loss, optimizer):
        self.loss = loss
        self.optimizer = optimizer

    def forward(self, inputs):
        for layer in self.layers:
            layer.forward(inputs)
            inputs = layer.output
        return inputs

    def backward(self, y_true):
        for i in reversed(range(len(self.layers))):
            if i == len(self.layers) - 1:
                self.layers[i].backward(self.layers[i].output, y_true)
            else:
                self.layers[i].backward(self.layers[i + 1].dinputs)

    def update_params(self):
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'weights'):
                self.optimizer.update_params(layer, i)

    # funkcija za učenje 
    def train(self, X_train, Y_train, epochs, batch_size):
        for epoch in range(epochs):
            for step in range(len(X_train) // batch_size):
                # mini baza za treniranje
                batch_X = X_train[step * batch_size:(step + 1) * batch_size]
                batch_y = Y_train[step * batch_size:(step + 1) * batch_size]
                
                # izračun naprej
                output = self.forward(batch_X)
                # izračunamo napako
                loss = self.loss.compute_mean_loss(output, batch_y)
                # izračun nazaj 
                self.backward(batch_y)
                self.update_params()

            print(f"Epoch {epoch + 1}, Loss: {loss}")
    
    # funkcija za validacijo na testni bazi
    def validate(self, X_val, Y_val):
        output = self.forward(X_val)
        loss = self.loss.compute_mean_loss(output, Y_val)
        predictions = np.argmax(output, axis=1)
        if len(Y_val.shape) == 2:
            Y_val = np.argmax(Y_val, axis=1)
        accuracy = np.mean(predictions == Y_val)
        return loss, accuracy


# Naložimo učne in testne podatke in oznake 
def load_dataset():
    df_train = pd.read_csv('isolet1+2+3+4.data', header=None)
    df_test = pd.read_csv('isolet5.data', header=None)

    scaler = StandardScaler()
    encoder = OneHotEncoder(categories='auto')
    
    X_train = scaler.fit_transform(df_train.iloc[:, :-1])
    Y_train = encoder.fit_transform(df_train.iloc[:, -1].values.reshape(-1, 1) - 1).toarray()
    X_test = scaler.transform(df_test.iloc[:, :-1])
    Y_test = encoder.transform(df_test.iloc[:, -1].values.reshape(-1, 1) - 1).toarray()

    return X_train, Y_train, X_test, Y_test

# Naložimo učne in testne vzorce 
X_train, Y_train, X_test, Y_test = load_dataset()

# Ustvarimo model, in definiramo strukturo omrežja
model = NeuralNetwork()
model.add(Layer(617, 44)) # vsak ISOLET primer ima 617 značilk,  število nevronov v skritem sloju
#model.add(SigmoidActivation())
model.add(ReLUActivation())  # definiramo katero aktivacijsko funkcijo želimo uporabiti
model.add(Layer(44, 26))  # izhodni sloj, število število vhodov in izhodov, 26 razredov
model.add(SoftmaxActivation()) # definiramo aktivacijsko funkcijo
model.set(CategoricalCrossentropyLoss(), MomentumOptimizer(learning_rate=0.01, momentum=0.9))

# Train the model
epochs = 40
batch_size = 128
model.train(X_train, Y_train, epochs, batch_size)

# Validate the model
val_loss, val_accuracy = model.validate(X_test, Y_test)
print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")
