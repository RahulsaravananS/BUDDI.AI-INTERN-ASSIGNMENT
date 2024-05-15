import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def gradient_descent(x, y, lr=0.01, test_size=0.2, converged=1e-6, random_state=None):
    # Splitting data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
    
    # Initial random values for coefficients
    b0 = np.random.normal(0, 1)
    b1 = np.random.normal(0, 1)
    
    # Calculate initial error
    error = np.mean((y_train - (b0 + b1 * np.array(x_train))) ** 2)
    
    # Initialize epoch, error, and coefficient
    epoch = 0
    Epoch = [0]
    E_train = [error]
    E_test = [np.mean((y_test - (b0 + b1 * np.array(x_test))) ** 2)]
    Gb0 = [b0]
    Gb1 = [b1]
    
    # Loop until convergence
    while True:
        # Compute gradients
        gradb0 = 0
        gradb1 = 0
        for i in range(len(x_train)):
            gradb0 = -2 * (y_train[i] - (b0 + b1 * x_train[i]))
            gradb1 = -2 * x_train[i] * (y_train[i] - (b0 + b1 * x_train[i]))
        
            # Update coefficients
            b0 -= lr * gradb0
            b1 -= lr * gradb1
            
            # Compute new error for training and testing sets
            new_error_train = np.mean((y_train - (b0 + b1 * np.array(x_train))) ** 2)
            new_error_test = np.mean((y_test - (b0 + b1 * np.array(x_test))) ** 2)
            epoch += 1
            
            # Store values for plotting and analysis
            Epoch.append(epoch)
            E_train.append(new_error_train)
            E_test.append(new_error_test)
            Gb0.append(b0)
            Gb1.append(b1)
            
            # Check convergence
            if abs(error - new_error_train) <= converged:
                break
            else:
                error = new_error_train
    
        return b0, b1, E_train, E_test, Epoch, Gb0, Gb1

# Generating random 1000 values to be appended in x
x = []
y = []
for i in range(-500, 500):
    x_val = i / 1000
    x.append(x_val)
    n = np.random.normal(0, 5)
    y_val = 2 * x_val - 3 + n
    y.append(y_val)

# Running gradient descent
b0_final, b1_final, Et, Et2, Epocht, Gb0t, Gb1t = gradient_descent(x, y, random_state=42)

# Printing final coefficients and error
print("Stochastic Gradient Descent: b0:", b0_final, "b1:", b1_final,"Epoch:",Epocht[-1])
print("Final Training Error:", Et[-1])
print("Final Testing Error:", Et2[-1])

# Plotting the error convergence
plt.figure(figsize=(8, 4))
plt.plot(Epocht, Et, label="Training Error")
plt.plot(Epocht, Et2, label="Testing Error")
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.title('Stochastic Gradient Descent')
plt.legend()
plt.figtext(0.5, 0.01, "In the above graph, the Mean Square Error of Training data and testing data of stochastic gradient descent is plotted with respect to the number of Epoch cycles executed", ha="center", fontsize=10, bbox={"facecolor":"brown", "alpha":0.5, "pad":5})

plt.show()
