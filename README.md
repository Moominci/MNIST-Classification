# MNIST-Classification
MNIST Classification using Custom Dataset, DataLoader



## Model Architecture Overview

### LeNet-5 Model
LeNet-5 is a pioneering convolutional network that includes three convolutional layers and two fully connected layers. It's designed for handwritten and machine-printed character recognition.

- **Total Parameters**: 61,706
- **Layers**:
  - Convolutional Layer 1: 156 parameters
  - Convolutional Layer 3: 2,416 parameters
  - Convolutional Layer 5: 48,120 parameters
  - Fully Connected Layer 6: 10,164 parameters
  - Output Layer: 850 parameters

### Custom MLP Model
Our custom MLP model is designed to match the parameter count of LeNet-5 closely, making it suitable for similar tasks with a fully connected architecture.

- **Total Parameters**: 64,630
- **Layers**:
  - First Layer: 62,880 parameters
  - Second Layer: 1,620 parameters
  - Third Layer: 230 parameters

Both models are implemented in PyTorch, providing robustness and flexibility for further modifications and testing.

### Comparison
The custom MLP model has a parameter count similar to LeNet-5, with a slightly higher number due to the dense connections in the MLP design. This similarity ensures that our MLP model maintains a comparable capacity and complexity.

For detailed implementation, please refer to `model.py` in this repository.
