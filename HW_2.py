# 第二周作业：尝试完成一个多分类任务的训练:一个随机向量，哪一维数字最大就属于第几类。

import torch.optim as optim
import torch
import torch.nn as nn
import numpy as np

# set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# define the model class
class NumberModel(nn.Module):
    def __init__(self, input_size, num_classes):
        # call the parent class constructor
        super(NumberModel, self).__init__()
        # define a linear layer: input size -> num_classes
        self.linear = nn.Linear(input_size, num_classes)
        # define loss function (use CrossEntropyLoss for multi-class classification)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        # pass the input through the linear layer to get the output logits
        x = self.linear(x)
        return x
    
# create random dataset for multi-class classification
def create_data(num_sample, num_classes):
    x = [] # list to store the input vectors
    y = [] # list to store the labels
    for i in range(num_sample):
        # randomly generate a vector of size num_classes
        vec = np.random.random(num_classes)
        # append the vector to the input list
        x.append(vec)
        # determine the label based on which dimension has the maximum value
        label = np.argmax(vec)
        # append the label to the label list
        y.append(label)
    # convert lists to numpy arrays and then to PyTorch tensors
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

# hyperparameters
input_dim = 5 # input dimension (number of features)
num_classes = 5 # number of classes (same as input dimension for this task)
num_sample = 1000 # number of samples in the training dataset

# create training and testing datasets
x_train, y_train = create_data(num_sample, num_classes)
x_test, y_test = create_data(200, num_classes)

# initialize the model, define the loss function and the optimizer
test_model = NumberModel(input_dim, num_classes)
# define the loss function
criterion = nn.CrossEntropyLoss()
# define the optimizer SGD
optimizer = optim.SGD(test_model.parameters(), lr=0.1)

# train loop
for epoch in range(100):
    # reset the gradients to zero
    optimizer.zero_grad()
    # forward pass: compute the output of the model
    output = test_model(x_train)
    # compute the loss between the output and the true labels
    loss = criterion(output, y_train)
    # backward pass: compute the gradients of the loss with respect to the model parameters
    loss.backward()
    # update the model parameters using the optimizer
    optimizer.step()
    # print the loss and accuracy every 20 epochs
    if (epoch +1) % 20 ==0:
        with torch.no_grad():
            pred = test_model(x_test).argmax()
            accuracy = (pred == y_test).float().mean()
            print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}, Accuracy: {accuracy.item():.4f}")

# showing output weights after training
print("\nafter training: (fc.weight):")
print(test_model.linear.weight.data)

# test the model on a new random vector
with torch.no_grad():
    # new random vector for testing
    test_vec = torch.tensor([0.1, 0.7, 0.9, 0.2, 0.5], dtype=torch.float32)
    output = test_model(test_vec)
    pred = output.argmax().item()
    # print the test vector, predicted class, and correct class
    print(f"\nTest vector: {test_vec.numpy()}")
    print(f"Predicted class: {pred}, correct class: {np.argmax(test_vec.numpy())}")

    