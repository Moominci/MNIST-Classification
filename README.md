# MNIST-Classification
MNIST Classification using Custom Dataset, DataLoader



## Model Architecture Overview

### LeNet-5 Model
LeNet-5 is a convolutional network that includes three convolutional layers and two fully connected layers.

- **Layers**:
  - Convolutional Layer 1: (5×5×1+1)×6 = 156 parameters
  - Convolutional Layer 3: (5×5×6+1)×16 = 2,416 parameters
  - Convolutional Layer 5: (5×5×16+1)×120 = 48,120 parameters
  - Fully Connected Layer 6: (120×84+84) = 10,164 parameters
  - Output Layer: (84×10+10) = 850 parameters
- **Total Parameters**: 61,706
- 
### Custom MLP Model
Our custom MLP model is designed to match the parameter count of LeNet-5 closely, making it suitable for similar tasks with a fully connected architecture.

- **Layers**:
  - First Layer: 784×80+80 = 62,880 parameters
  - Second Layer: 80×20+20 = 1,620 parameters
  - Third Layer: 20×10+10 = 230 parameters
- **Total Parameters**: 64,630

Both models are implemented in PyTorch.

### Comparison
The custom MLP model has a parameter count similar to LeNet-5, with a slightly higher number due to the dense connections in the MLP design. This similarity ensures that our MLP model maintains a comparable capacity and complexity.

For detailed implementation, please refer to `model.py` in this repository.
