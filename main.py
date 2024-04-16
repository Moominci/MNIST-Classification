import dataset
from model import LeNet5, CustomMLP
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt


# import some packages you need here


def train(model, train_loader, device, criterion, optimizer):
    """ Train function

    Args:
        model: network
        trn_loader: torch.utils.data.DataLoader instance for training
        device: device for computing, cpu or gpu
        criterion: cost function
        optimizer: optimization method, refer to torch.optim

    Returns:
        trn_loss: average loss value
        acc: accuracy
    """
    model.train()
    total_loss = 0
    correct = 0
    for imgs, labels in tqdm(train_loader, desc="Training", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / len(train_loader.dataset)
    print(f"Train Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        
    # write your codes here

    return total_loss, accuracy

def test(model, test_loader, device, criterion):
    """ Test function

    Args:
        model: network
        tst_loader: torch.utils.data.DataLoader instance for testing
        device: device for computing, cpu or gpu
        criterion: cost function

    Returns:
        tst_loss: average loss value
        acc: accuracy
    """

    # write your codes here
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Testing", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    avg_loss = total_loss / len(test_loader)
    accuracy = 100 * correct / len(test_loader.dataset)
    print(f"Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    return total_loss, accuracy

def plot_statistics(statistics, title, ylabel, filename):
    plt.figure(figsize=(10, 5))
    plt.plot(statistics)
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel(ylabel)
    plt.savefig(filename)
    plt.close()
    
def main():
    """ Main function

        Here, you should instantiate
        1) Dataset objects for training and test datasets
        2) DataLoaders for training and testing
        3) model
        4) optimizer: SGD with initial learning rate 0.01 and momentum 0.9
        5) cost function: use torch.nn.CrossEntropyLoss

    """

    # write your codes here
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    train_loader, test_loader = dataset.get_loader(train_df='train_df.csv', test_df='test_df.csv', batch_size=64, test_batch_size=64)    
    
    models = {'LeNet5': LeNet5(), 'CustomMLP': CustomMLP()}
    for name, model in models.items():
        model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        # optimizer_L2 = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.001)  # Apply L2 regularization

        train_losses, train_accuracies = [], []
        test_losses, test_accuracies = [], []
        
        epochs = 15
        for epoch in range(epochs):
            print(f"{name} Epoch {epoch+1}/{epochs}")
            train_loss, train_accuracy = train(model, train_loader, device, criterion, optimizer)
            # train_loss, train_accuracy = train(model, train_loader, device, criterion, optimizer_L2) # Apply L2 regularization
            test_loss, test_accuracy = test(model, test_loader, device, criterion)
            
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            test_losses.append(test_loss)
            test_accuracies.append(test_accuracy)

        # Plotting results
        plot_statistics(train_losses, f'{name} Training Loss', 'Loss', f'{name}_train_loss.png')
        plot_statistics(train_accuracies, f'{name} Training Accuracy', 'Accuracy (%)', f'{name}_train_accuracy.png')
        plot_statistics(test_losses, f'{name} Testing Loss', 'Loss', f'{name}_test_loss.png')
        plot_statistics(test_accuracies, f'{name} Testing Accuracy', 'Accuracy (%)', f'{name}_test_accuracy.png')


if __name__ == '__main__':
    main()
