# Neural Network from Scratch in NumPy

## Motivation
I built this project to actually understand what's happening inside a neural network under the hood. The goal was simple: implement everything from scratch using only raw Python, NumPy, and math—no PyTorch, no Keras, no magic. 

By training a custom neural network on the MNIST handwritten digit dataset, I wanted to gain an intuitive understanding of the forward pass, backpropagation, and gradient descent. More importantly, I intentionally broke the model in several ways to better understand common pitfalls like bad learning rates, poor weight initialization, and the vanishing gradient problem.

## The Process
Building the network went step-by-step:
1. **Data Preprocessing:** Flattening the 28x28 MNIST images into 784-dimensional vectors and applying one-hot encoding to the labels.
2. **Forward Pass:** Building linear transformations and passing them through activation functions (ReLU, Sigmoid, Softmax).
3. **Loss Function:** Implementing Categorical Cross-Entropy to optimize for classification.
4. **Backpropagation:** Manually deriving and writing the chain rule steps traversing backward through the network to calculate gradients.
5. **Optimization:** Updating weights and biases using Mini-batch Gradient Descent.

## What I Tried & What Failed
One of the most valuable parts of this project was experimenting with different setups to see how the network would fail. 

### Experiment 1: Learning Rate Extremes
- **What I tried:** I initially set the learning rate high (`lr = 0.1` and `lr = 10.0`). 
- **What happened:** With a very high learning rate, the loss immediately exploded, overshooting the minimum every step and failing to converge.
- **The Fix:** I reduced the learning rate for stability (`lr = 0.01`). Later, I implemented **Learning Rate Scheduling** (`start_lr=0.1` with `decay=0.92` per epoch), which provided fast early progress and careful fine-tuning toward the end.

### Experiment 2: Bad Weight Initialization
- **What I tried:** I initialized weights to completely zeros, and then tried large random values.
- **What happened (Zero Init):** Every neuron computed the same output, received identical gradients, and updated identically. The network suffered from the **symmetry problem** and failed to learn different features.
- **What happened (Large Init):** Activations saturated immediately. Most neurons using ReLU were stuck outputting 0. Gradients became zero, and nothing updated.
- **The Fix:** Using **He Initialization**, which specifically scales weights by `sqrt(2/n_inputs)`. This keeps the variance of activations stable as data flows through the layers.

### Experiment 3: Activation Functions and Vanishing Gradients
- **What I tried:** Initially, I used the **Sigmoid** activation function for my hidden layers.
- **What happened:** In a deep network setup, the gradients essentially vanished before they could reach the earlier layers (because the maximum derivative of a Sigmoid is 0.25). A derivative of `0.25^8 = 0.000015` meant that my early layers completely stopped learning anything useful.
- **The Fix:** Switching to **ReLU**. Since ReLU's derivative is either 0 or 1, the gradients maintain their magnitude, allowing even deep networks to learn effectively.

### Other Issues Faced
- **Mean Squared Error (MSE):** I first tried using MSE for calculating loss, but it performed poorly for a classification task. Switching to Cross-Entropy drastically improved results.
- **Shape Mismatches & Backprop Coding:** Figuring out dimensional alignment, particularly handling transposes during backprop, caused errors multiple times. Doing it manually forced me to carefully track matrix math.
- **Bias:** I initially forgot to implement bias terms, which degraded model performance significantly until fixed.

## Future Improvements / Limitations
While the baseline model achieves ~95% test accuracy on MNIST, it struggles with:
- Ambiguous digits (e.g., 4 vs. 9, 3 vs. 8).
- Lack of Spatial Awareness: Fully connected layers treat pixel 1 and pixel 783 as equally related, ignoring the actual 2D spatial structure of an image.
Pushing past 98% accuracy would require adding Convolutional (Conv2D) layers, Dropout, and Batch Normalization.
