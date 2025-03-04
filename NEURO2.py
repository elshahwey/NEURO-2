#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
# Sigmoid activation
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x):
    return x * (1 - x)
# Training data (XOR problem)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])
# Initialize weights and biases
np.random.seed(1)
input_size = 2
hidden_size = 2
output_size = 1
weights_input_hidden = np.random.uniform(-1, 1, (input_size, hidden_size))
weights_hidden_output = np.random.uniform(-1, 1, (hidden_size, output_size))
bias_hidden = np.zeros((1, hidden_size))
bias_output = np.zeros((1, output_size))
# Training parameters
learning_rate = 0.5
epochs = 10000
# Training loop
for epoch in range(epochs):
    # Forward propagation
    hidden_layer_input = np.dot(X, weights_input_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)
    
    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    output = sigmoid(output_layer_input)
    # Compute error
    error = y - output
    # Backpropagation
    output_gradient = error * sigmoid_derivative(output)
    hidden_error = output_gradient.dot(weights_hidden_output.T)
    hidden_gradient = hidden_error * sigmoid_derivative(hidden_layer_output)
    # Update weights and biases
    weights_hidden_output += hidden_layer_output.T.dot(output_gradient) * learning_rate
    weights_input_hidden += X.T.dot(hidden_gradient) * learning_rate
    bias_output += np.sum(output_gradient, axis=0, keepdims=True) * learning_rate
    bias_hidden += np.sum(hidden_gradient, axis=0, keepdims=True) * learning_rate
    # Print error at intervals
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Error: {np.mean(np.abs(error))}")
# Final output after training
print("Final output after training:")
print(output)


# In[ ]:




