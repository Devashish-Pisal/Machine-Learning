from sklearn.datasets import fetch_california_housing
import torch
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
import torchviz
from torchsummary import summary

# Load the dataset
housing = fetch_california_housing()


# Columns in dataset:
# ['MedInc' (0), 'HouseAge' (1), 'AveRooms' (2), 'AveBedrms' (3), 'Population' (4), 'AveOccup' (5), 'Latitude' (6), 'Longitude' (7)]

# Split the data training and testing
X_train, X_test, Y_train, Y_test = train_test_split(housing.data, housing.target, test_size=0.2, random_state=42)

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert the data to tensors for fast computations
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
Y_train = torch.tensor(Y_train, dtype=torch.float32).view(-1,1)
Y_test = torch.tensor(Y_test, dtype=torch.float32).view(-1,1)

# Define the model class
class LinearRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(in_features=8, out_features=1)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

# Create model instance
model = LinearRegressionModel()

# Define loss function and optimizer
loss_function = torch.nn.MSELoss(size_average = True) # Mean squared error
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01) # Stochastic Gradient Descent

# Do computations on cuda (if installed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_train = X_train.to(device)
X_test = X_test.to(device)
Y_train = Y_train.to(device)
Y_test = Y_test.to(device)
model = model.to(device)


# Store the training data
epochs = 1000
training_loss = np.zeros(epochs)

# Train the model
model.train()
for epoch in range(epochs):
    y_pred = model(X_train)
    loss = loss_function(y_pred, Y_train)
    print(f'Training Loss for epoch {epoch + 1}: {loss.item():.2F}')
    training_loss[epoch] = loss.item()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Plot training loss
plt.scatter(range(1,epochs+1), training_loss, s=5)
plt.ylabel('Training Loss')
plt.xlabel('Epochs')
plt.title('Training Loss vs Epochs')
plt.show()

# Testing the trained model
model.eval()
with torch.no_grad():
    y_pred = model(X_test)
test_loss = loss_function(y_pred, Y_test)
print("=================================================")
print(f'Test Loss: {test_loss.item():.2F}')
print("=================================================")



# Visualize testing data
plt.scatter(Y_test.cpu(), y_pred.cpu(), s=5, color='green')
plt.plot([Y_test.cpu().min(), Y_test.cpu().max()], [Y_test.cpu().min(), Y_test.cpu().max()], 'r--')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Predicted vs Actual Values')
plt.show()


# Visualize the backpropagation control flow
dummy_input = torch.randn(1, 8).to(device)
torchviz.make_dot(model(dummy_input), params=dict(model.named_parameters())).render("linear_regression_backward-pass", format="png")

# Print the model summary
print(summary(model, (8,)))