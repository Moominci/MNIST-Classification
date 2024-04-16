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


## Performance Analysis Report

This report analyzes the performance of two neural network models: the LeNet-5 and a custom Multi-Layer Perceptron (MLP). The performance is measured in terms of accuracy and loss on both the training and testing (used as validation) datasets.

### LeNet-5 Model Performance

#### Training Performance
- Training Accuracy: Shows a similar trend to the Custom MLP, with a quick rise and high stabilization.
- Training Loss: Decreases consistently, indicating a strong fit to the training data.

#### Testing Performance
- Testing Accuracy: Remains consistently high, with a peak testing accuracy of approximately 99%, which is characteristic of LeNet-5's performance.
- Testing Loss: Shows a downward trend with some fluctuation, typical in the testing phase due to the varied nature of the test samples.

<img src="./plot_images/no_aug_normal_opt_epochs_15/LeNet5_train_accuracy.png" width="650" height="300" alt="LeNet-5 Training Accuracy">
<img src="./plot_images/no_aug_normal_opt_epochs_15/LeNet5_train_loss.png" width="650" height="300" alt="LeNet-5 Training Loss">
<img src="./plot_images/no_aug_normal_opt_epochs_15/LeNet5_test_accuracy.png" width="650" height="300" alt="LeNet-5 Testing Accuracy">
<img src="./plot_images/no_aug_normal_opt_epochs_15/LeNet5_test_loss.png" width="650" height="300" alt="LeNet-5 Testing Loss">


### Custom MLP Model Performance

The Custom MLP model is designed with a similar number of parameters as the LeNet-5 model, and its performance is measured on the MNIST dataset.

#### Training Performance
- Training Accuracy: The model's accuracy on the training set shows a steep increase initially, stabilizing at high values, indicating effective learning.
- Training Loss: The loss decreases rapidly, suggesting that the model is fitting well to the training data.

#### Testing Performance
- Testing Accuracy: Accuracy on the testing set improves consistently, with slight fluctuations, peaking around 97.5%. This suggests good generalization.
- Testing Loss: The testing loss decreases and then fluctuates, which is expected as the model encounters various complexities within the test data.

<img src="./plot_images/no_aug_normal_opt_epochs_15/CustomMLP_train_accuracy.png" width="650" height="300" alt="Custom MLP Training Accuracy">
<img src="./plot_images/no_aug_normal_opt_epochs_15/CustomMLP_train_loss.png" width="650" height="300" alt="Custom MLP Training Loss">
<img src="./plot_images/no_aug_normal_opt_epochs_15/CustomMLP_test_accuracy.png" width="650" height="300" alt="Custom MLP Testing Accuracy">
<img src="./plot_images/no_aug_normal_opt_epochs_15/CustomMLP_test_loss.png" width="650" height="300" alt="Custom MLP Testing Loss">


## Comparative Performance Analysis: LeNet-5 vs Custom MLP

This section presents a comparison between the predictive performances of the LeNet-5 and the custom MLP models trained on the same dataset.

### Performance Metrics
evaluate the models based on two primary metrics:
1. **Accuracy**: The proportion of true results (both true positives and true negatives) among the total number of cases examined.
2. **Loss**: A numerical representation of how far the model's predictions are from the actual labels.

### LeNet-5 Model Performance
The LeNet-5 model demonstrates excellent predictive performance with the testing accuracy peaking around 98.9%. This is in line with historical performance metrics known for LeNet-5 on similar datasets. The testing loss for LeNet-5 also shows fluctuations, which is common in testing scenarios due to the model encountering varied difficulty levels within the test data.

### Custom MLP Model Performance
The Custom MLP model shows a steady increase in testing accuracy, reaching approximately 97.5% by the final epoch. This demonstrates strong predictive performance. However, there is some fluctuation in testing loss, indicating variability in the model's predictions on different batches of the test set.


### Comparison and Insights
When comparing the two models:
- **LeNet-5** tends to have a slightly higher **testing accuracy** than the custom MLP, which aligns with expectations given its convolutional structure that is specifically advantageous for image recognition tasks.
- **Testing loss** for both models shows some volatility, which is typical in practice due to the variability inherent in test data.

Overall, both models perform competitively, with LeNet-5 having a slight edge in accuracy. This might be attributed to LeNet-5's convolutional layers, which are adept at capturing spatial hierarchies in image data.

The observed accuracies for both models are similar to known benchmarks, indicating that the training processes were successful and the models are well-tuned for the task.

In conclusion, both the LeNet-5 and custom MLP models demonstrate strong capabilities in recognizing handwritten digits with LeNet-5 showing a slight advantage in overall accuracy.


## Employ regularization techniques 

used three regularization techniques
- augmentation : RandomRotation, RandomAffine (Rotate images, Moving the images from left to right)
- dropout : Dropout(0.25)
- L2 normalization : optim.weight_decay
