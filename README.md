## Neural Network Training on MNIST

This neural network is trained on the well-known **MNIST dataset**, consisting of:

- **60,000 training images**
- **10,000 test images**

### Training Approaches

Initially, the network was trained using **Stochastic Gradient Descent (SGD)**. While effective and fast (approximately **12 minutes** training time), it left room for improvement in terms of accuracy.

To enhance performance, the training was later switched to **Mini-Batch Gradient Descent**, which:

- Enabled more efficient and stable updates of the **biases**
- Increased the final test accuracy to **97.3%**
- Increased training time to approximately **90 minutes**

> **Note:** The extended training time is due to the network being implemented **without deep learning frameworks** such as PyTorch or TensorFlow. As a result, the training process **cannot be accelerated via GPU or TPU**.
