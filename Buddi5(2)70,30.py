# Import necessary libraries or modules
import numpy as np
import matplotlib.pyplot as plt
import random
# Initialize the x matrix with a column of ones and empty lists for other powers of x
x = [[1] * 101, [], [], [], []]
x1 = []  # List to store x values
y = []   # List to store y values

# Generate x values in the range -5 to 5
for i in range(-50, 51):
    i = i / 10
    x1.append(i)
    # Append x values for different powers
    x[1].append(i)  # Linear
    x[2].append(i ** 2)  # Quadratic
    x[3].append(i ** 3)  # Cubic
    x[4].append(i ** 4)  # Biquadratic
    # Generate y values with some noise
    n = np.random.normal(0, 3)
    fx = ((2 * (i ** 4)) - (3 * (i ** 3)) + (7 * (i ** 2)) - (23 * i) + 8 + n)
    y.append(fx)

# Transpose x and y for matrix operations
X = np.transpose(x)
Y = np.transpose(y)

# Calculate coefficients using matrix operations
b = np.matmul(np.linalg.inv(np.matmul(np.transpose(X), X)), np.matmul(np.transpose(X), y))

# Generate predicted y values for each model
y1 = b[0] + b[1] * np.array(x1)  # Linear
y2 = b[0] + b[1] * np.array(x1) + b[2] * np.array(x1) ** 2  # Quadratic
y3 = b[0] + b[1] * np.array(x1) + b[2] * np.array(x1) ** 2 + b[3] * np.array(x1) ** 3  # Cubic
y4 = b[0] + b[1] * np.array(x1) + b[2] * np.array(x1) ** 2 + b[3] * np.array(x1) ** 3 + b[4] * np.array(x1) ** 4  # Biquadratic

# Lagrange Interpolation function
def lagrangeInterpolation(x, y, xInterp):
    n = len(x)
    m = len(xInterp)
    yInterp = np.zeros(m)
    
    for j in range(m):
        p = 0
        for i in range(n):
            L = 1
            for k in range(n):
                if k != i:
                    L *= (xInterp[j] - x[k]) / (x[i] - x[k])
            p += y[i] * L
        yInterp[j] = p
    return yInterp

# Calculate interpolated y values using Lagrange Interpolation
yInte = lagrangeInterpolation(x1, y, x1)

# Plot the models and Lagrange interpolation
plt.figure(figsize=(8, 4))
plt.plot(x1, y1, label="linear")
plt.plot(x1, y2, label="quadratic")
plt.plot(x1, y3, label="cubic")
plt.plot(x1, y4, label="biquadratic")
plt.plot(x1, yInte, marker='o', label="lagrange")
plt.xlabel('X')
plt.ylabel('Y=F(X)')
plt.title("Different models and their plot")
plt.figtext(0.5, 0.01, "In this graph, 101 x values in the range (-5, 5) and generated y values for different models such as linear, quadratic, cubic, biquadratic, and Lagrange are plotted.", ha="center", fontsize=10, bbox={"facecolor": "brown", "alpha": 0.5, "pad": 5})
plt.legend()
plt.show()

# Define model complexity
com=[1,2,3,4]
# random.shuffle(x1)
# Splitting data for training and testing
X1 = x1[:71]  # Training data
X2 = x1[71:]  # Testing data
Yorig = Y[71:]  # Original Y values for testing

# Predicted values for training data
y1Train = b[0] + b[1] * np.array(X1)  # Linear
y2Train = b[0] + b[1] * np.array(X1) + b[2] * np.array(X1) ** 2  # Quadratic
y3Train = b[0] + b[1] * np.array(X1) + b[2] * np.array(X1) ** 2 + b[3] * np.array(X1) ** 3  # Cubic
y4Train = b[0] + b[1] * np.array(X1) + b[2] * np.array(X1) ** 2 + b[3] * np.array(X1) ** 3 + b[4] * np.array(X1) ** 4  # Biquadratic

# Calculate training errors
trainError1 = np.square(np.abs(y1Train - Y[:71]))
trainError2 = np.square(np.abs(y2Train - Y[:71]))
trainError3 = np.square(np.abs(y3Train - Y[:71]))
trainError4 = np.square(np.abs(y4Train - Y[:71]))
totalTrainError = [np.sum(trainError1)/len(trainError1), np.sum(trainError2)/len(trainError2), np.sum(trainError3)/len(trainError3), np.sum(trainError4)/len(trainError4)]

# Predicted values for testing data
y1Test = b[0] + b[1] * np.array(X2)  # Linear
y2Test = b[0] + b[1] * np.array(X2) + b[2] * np.array(X2) ** 2  # Quadratic
y3Test = b[0] + b[1] * np.array(X2) + b[2] * np.array(X2) ** 2 + b[3] * np.array(X2) ** 3  # Cubic
y4Test = b[0] + b[1] * np.array(X2) + b[2] * np.array(X2) ** 2 + b[3] * np.array(X2) ** 3 + b[4] * np.array(X2) ** 4  # Biquadratic

# Calculate testing errors
testError1 = np.square(np.abs(y1Test - Yorig))
testError2 = np.square(np.abs(y2Test - Yorig))
testError3 = np.square(np.abs(y3Test - Yorig))
testError4 = np.square(np.abs(y4Test - Yorig))
totalTest = [np.sum(testError1)/len(testError1), np.sum(testError2)/len(testError2), np.sum(testError3)/len(testError3), np.sum(testError4)/len(testError4)]

# Plot Bias-Variance tradeoff
plt.figure(figsize=(8, 4))
plt.plot(com, totalTest, label="Variance")
plt.plot(com, totalTrainError, label="Bias")
plt.xlabel('Model Complexity')
plt.ylabel('Error')
plt.title("Bias-Variance Tradeoff")
plt.figtext(0.5, 0.01, "The above graph shows the bias-variance tradeoff for models such as linear, quadratic, cubic, and biquadratic models", ha="center", fontsize=10, bbox={"facecolor": "brown", "alpha": 0.5, "pad": 5})
plt.legend()
plt.show()
