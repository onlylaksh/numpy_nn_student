# Building Neural Network from Scratch using NumPy

## What is this?

I had to build a network for my machine learning course. without using PyTorch I decided to build it from scratch with just NumPy. I did not fully understand backpropagation until I had to code it myself. The assignment was to get than 90 percent on MNIST. I got 94.8 percent, which is way above what they asked for so I am pretty happy with it.

## Whats in the notebook

I organized it into sections so my teaching assistant can follow along:

1. **Imports**. I used numpy, matplotlib and tensorflow to load MNIST. I promise I did not use tensorflow for anything

2. **Load MNIST**. I flattened the 28x28 images into 784-length vectors normalized pixel values to 0-1 and one-hot encoded the labels.

3. **Activation functions**. I used ReLU, Sigmoid and Softmax and also their derivatives for backprop.

4. **Weight initialization**. I used He init, zero init and a "bad" large init for the experiments.

5. **Forward pass**. I did matrix multiplication activation, layer by layer.

6. **Loss function**. I used cross entropy. I had to add clipping because log(0) gave me NaN for like 2 hours.

7. **Backpropagation**. This took me forever to get right. The chain rule is confusing when you actually have to code it.

8. **Training loop**. I used mini- gradient descent.

9. **Baseline model**. I used 784 → 128 → 64 → 10 He init, ReLU and learning rate equals 0.01.

10. **Experiment 1: Learning rate stuff**. I tried small versus too large and learning rate scheduling.

11. **Experiment 2: Weight initialization**. I tried zero init. Large init so you can see why He init is good.

12. **Experiment 3: gradients**. I tried a deep network with sigmoid versus ReLU.

13. **Comparison plots**. I put all experiments together so you can see the differences.

14. **Failure cases**. I looked at images the neural network model got wrong. Some of them are actually illegible.

15. **What I learned**. These are my takeaways from this thing.


## The experiments

### Experiment 1. Learning Rate

- **lr equals 0.00001 (too small)**: Loss barely decreases. The neural network model is too slow to learn anything in 30 epochs.

- **lr equals 10.0 ( large)**: Loss goes up and down wildly sometimes becomes NaN when it overflows.

- **Fix**: Learning rate scheduling. I started at 0.1 multiplied by 0.92 after each epoch. This let me start aggressive and then take steps as I got closer to the minimum.

### Experiment 2. Weight Initialization

- **Zero init**: Every neuron learns the thing because they all start identical. Gradients are identical so the neural network might well have 1 hidden neuron. Accuracy is basically guessing.

- **Large init**: Weights start huge so activations are saturated. Sigmoid outputs are either 0 or 1 gradients are tiny nothing learns.

- **Fix**: He initialization. I scaled weights by sqrt(2/fan_in). This kept variance across layers so gradients did not explode or vanish.

### Experiment 3. Vanishing Gradients

- ** sigmoid network**: 5 hidden layers with sigmoid. Gradient at layers is (sigmoid_derivative)^depth. 0.25^8 Equals 0.000015. Basically no learning happens in few layers.

- **Deep ReLU network**: architecture but with ReLU. Gradient derivative is either 0 or 1 for the part. Better gradient flow through the neural network.

- **Fix**: Just use ReLU for layers. Sigmoid is only useful at the output layer for classification.


## How to run it

Open the notebook in Google Colab or Jupyter. Run cells in order. The baseline trains first then the experiments run after.

Requirements:

- numpy

- matplotlib

- tensorflow ( just for loading MNIST, nothing else)

Google Colab has all these preinstalled so just open it there and it should work.

## Results

| Model | Test Accuracy |

| Baseline (He, ReLU lr equals 0.01) | 94.8 percent |

| Bad LR ( small) | approximately 6 percent |

| Bad LR (too large) | unstable/NAN |

| With LR scheduling | approximately 95 percent |

| Zero initialization | 10 percent |

| Large initialization | approximately 11 percent |

| Deep network with Sigmoid | approximately 12 percent |

| Deep network with ReLU | 94 percent |


## How I actually built this (not the clean version)

### Where I started

My first attempt was a disaster. I wrote everything in one loop and when I ran it I got NaN for loss on the first epoch. I spent like 2 hours debugging. It turned out I was not clipping predictions in cross entropy so log(0) was happening. After that I refactored everything into functions which made debugging so much easier.

### Backpropagation struggles

This was the part. My first backprop implementation had the dimensions wrong for the weight gradient. I had `dZ @ A_prev` when it should have been `A_prev.T @ dZ`. The shapes worked because I was testing with matrices but it was actually wrong. I only caught it when I sat down with paper and drew out the computation graph for a neural network and computed the gradients manually.

Also my first backprop only worked for 3 layers. Making it work for any depth with a loop took another hour because I kept having off-by-one errors with the indices.

### Things I tried that did not work

I tried adding momentum before I tried learning rate scheduling. My loss got worse because I did not adjust the learning rate down when adding momentum. I scrapped it. Just did scheduling which was simpler and worked fine.

Batch size of 32 gave noisy loss curves and training took forever. I switched to 256. It was way smoother. Probably smaller batches would help with generalization. I just wanted stable training for the experiments.

I tried adding a hidden layer (256 neurons) to the baseline. Accuracy went from 94.8 percent to 95.1 percent. Training was twice as slow. Not worth it for 0.3 percent improvement.

### Vanishing gradient experiment

I honestly did not think it would be that bad. When I saw the deep sigmoid network stuck at 10 percent accuracy for 10 epochs I thought I broke something. I reran it twice before I did the math. Realized the early layers were getting literally no gradient signal. The ReLU version trained fine which was satisfying to see.

### What the numbers actually mean

94.8 percent on MNIST sounds good. That is still approximately 500 wrong predictions out of 10,000 test images. I looked at the misclassified ones and some of them are genuinely illegible. I would get them wrong too.. Some are clearly wrong in ways a human would not be which tells me the neural network model is missing spatial information. A CNN would fix this. That was not the assignment.

### What I would do differently

I should have written unit tests for each function before integrating. Would have caught the dimension bug in backprop way earlier.

Every epoch I recompute accuracy on the full training set which's 60,000 images. That is slow. I could have just tracked batch loss instead.

I did not use a validation set. I tuned hyperparameters on the test set which means my 94.8 percent is probably optimistic. A proper experiment would split off validation data.

### Time breakdown

- Forward pass plus loss: 2 hours

- Backprop and debugging dimensions: approximately 4 hours

- Generalizing to arbitrary depth: approximately 1 hour

- Three experiments: approximately 3 hours total

- Plots and writing: approximately 2 hours

Total maybe 12-13 hours over a few days. My roommate thought I was crazy, for not using PyTorch. I actually understand backprop now so worth it I guess.