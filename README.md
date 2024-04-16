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


## Neural Network Training Report

This report summarizes the training process of two neural network models: the LeNet-5 model. and a custom Multi-Layer Perceptron (MLP). We present the loss and accuracy metrics, both for training and testing datasets, across various epochs.

Below are the accuracy and loss plots derived from its training and testing processes.

### LeNet-5 Model

#### Training Accuracy and Loss
![LeNet5 Training Accuracy](LeNet5_train_accuracy.png)
- **Training Accuracy**: Depicts the proportion of correctly identified images by the model in the training set. An upward trend in accuracy is noted as training progresses.

![LeNet5 Training Loss](LeNet5_train_loss.png)
- **Training Loss**: Showcases the model's training error. The plot shows a steady decrease in loss, signifying consistent learning across epochs.

#### Testing Accuracy and Loss
![LeNet5 Testing Accuracy](LeNet5_test_accuracy.png)
- **Testing Accuracy**: Reflects the model's accuracy on the test dataset. This measure helps in understanding how well the model can generalize.

![LeNet5 Testing Loss](LeNet5_test_loss.png)
- **Testing Loss**: Depicts the test error for the model. Like the training loss, a downward trend would indicate improved performance, whereas fluctuations may point to overfitting or instability during training.

### Custom MLP Model

#### Training Accuracy and Loss
![Custom MLP Training Accuracy](CustomMLP_train_accuracy.png)
- **Training Accuracy**: Shows the percentage of correctly classified images in the training set across epochs. The model exhibits an increasing trend, indicating learning over time.

![Custom MLP Training Loss](CustomMLP_train_loss.png)
- **Training Loss**: Represents the model's error on the training set. A declining trend is observable, which is desirable as it indicates better model performance with each epoch.

#### Testing Accuracy and Loss
![Custom MLP Testing Accuracy](CustomMLP_test_accuracy.png)
- **Testing Accuracy**: Displays the model's performance on the test set. The plot demonstrates the model's generalization capabilities with new, unseen data.

![Custom MLP Testing Loss](CustomMLP_test_loss.png)
- **Testing Loss**: Illustrates the loss on the test set. Fluctuations suggest variations in model predictions against the actual labels in the test set.


## Conclusion

The plots above demonstrate the learning curve of both models through their respective training and testing phases. They serve as a visual tool to assess the model's performance, stability, and generalization capabilities over time.


