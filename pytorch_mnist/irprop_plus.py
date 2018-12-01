import torch
import torch.nn  as nn
import torch.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable
from Rprop_minus import *

# There are many datasets available in torchvision,
# one of them is MNIST, We have to convert it to
# a tensor using torchvision.transforms

train_dataset = dsets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = dsets.MNIST(root='./data', train=False,  transform=transforms.ToTensor(), download=True)

class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        #self.linear = nn.Linear(input_dim, output_dim)
        """ three hidden layers"""
        """
        self.fc1 = nn.Linear(28 * 28, 500)
        self.fc2 = nn.Linear(500, 200)
        self.fc3 = nn.Linear(200, 10)
        self.classifier = nn.Sequential(self.fc1, nn.ReLU(), self.fc2, nn.ReLU(), self.fc3)

        #self.classifier = nn.Sequential(self.fc1, nn.ReLU(), self.fc2)
        #self.classifier = nn.Linear(28*28, 10)
        """
        self.fc1 = nn.Linear(28 * 28, 500)
        self.fc2 = nn.Linear(500, 200)
        self.fc3 = nn.Linear(200, 10)
        self.classifier = nn.Sequential(self.fc1, nn.ReLU(), self.fc2, nn.ReLU(), self.fc3)

    def forward(self, x):
        x = self.classifier(x)

        return x


#training_set = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True)
#test_set = torch.utils.data.DataLoader(test_dataset, batch_size=100)
training_set = torch.utils.data.DataLoader(train_dataset, batch_size= len(train_dataset), shuffle=True)
test_set = torch.utils.data.DataLoader(test_dataset, batch_size= len(test_dataset))

input_dimensions = 784
output_dimensions = 10
model = LogisticRegression(input_dimensions, output_dimensions)
# Declare a loss criteria
criterion = nn.CrossEntropyLoss()
# define a learning rate
learning_rate = 0.001
# declare a optimizer, SGD(stochastic gradient descent)

#optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
#optimizer = torch.optim.RMSprop(model.parameters(), lr = 0.001)
optimizer = Rprop_minus(model.parameters(), lr = 0.001)


n_epochs = 50
iteration_no = 0
for epoch in range(n_epochs):
   for i, (images, labels) in enumerate(training_set):

        #clear the previous gradient
        optimizer.zero_grad()

        #Convert the images to a Tensor,for
        #calculating gradient
        #images.view creates 784dim column Tensor
        images = Variable(images.view(-1, 784))
        lables = Variable(labels)

        #forward pass
        output = model(images)
        #find the error/loss wrt true labels
        loss = criterion(output, labels)
        #back-prop
        loss.backward()
        #update the parameters
        if (iteration_no > 0 and float(prev_loss) < float(loss)):
            optimizer = Rprop(model.parameters(), lr = 0.001)

        optimizer.step()
        iteration_no +=1
        prev_loss = loss

        #testing - For checking the accuracy
        if(iteration_no%1 ==0):
            correct = 0
            total = 0
            for (test_images, labels) in test_set:
              #same process as training
                images = Variable(test_images.view(-1, 784))
                labels = Variable(labels)
                output = model(images)
                _, predicted = torch.max(output.data, 1)
                correct += (predicted == labels).sum()
                total += labels.size(0)
                accuracy = 100*correct/total
            print(f' Iteration: {iteration_no}, loss: {loss}, accuracy ={accuracy}')
