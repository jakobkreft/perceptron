import numpy as np

class Layer:
    def __init__(self, n_inputs, n_outputs):
        self.weights = np.random.randn(n_inputs, n_outputs) * 0.1
        self.biases = np.zeros((1, n_outputs))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)

class ReLUActivation:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

class SoftmaxActivation:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    def backward(self, dvalues, y_true):
        self.dinputs = dvalues.copy()
        self.dinputs[range(len(dvalues)), y_true.argmax(axis=1)] -= 1
        self.dinputs = self.dinputs / len(dvalues)

class CategoricalCrossentropyLoss:
    def compute_loss(self, y_pred, y_true):
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        correct_confidences = y_pred_clipped[range(len(y_pred)), y_true.argmax(axis=1)]
        return -np.log(correct_confidences)

    def compute_mean_loss(self, y_pred, y_true):
        return np.mean(self.compute_loss(y_pred, y_true))

class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.loss_function = CategoricalCrossentropyLoss()

    def add(self, layer):
        self.layers.append(layer)

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

    def train(self, X_train, Y_train, epochs, learning_rate):
        for epoch in range(epochs):
            output = self.forward(X_train)
            loss = self.loss_function.compute_mean_loss(output, Y_train)
            self.backward(Y_train)
            
            # Update weights and biases
            for layer in self.layers:
                if hasattr(layer, 'weights'):
                    layer.weights -= learning_rate * layer.dweights
                    layer.biases -= learning_rate * layer.dbiases

            print(f"Epoch {epoch + 1}, Loss: {loss}")

    def predict(self, X):
        output = self.forward(X)
        return output

# XOR Problem Data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])

# Create model
model = NeuralNetwork()
model.add(Layer(2, 3))   # 3 neurons in hidden layer
model.add(ReLUActivation())
model.add(Layer(3, 2))   # 2 output neurons (for 0 and 1)
model.add(SoftmaxActivation())

# Train the model
model.train(X, Y, epochs=1000, learning_rate=0.1)

# Predict and print probabilities
predictions = model.predict(X)
print("Probabilities:")
print(predictions)

