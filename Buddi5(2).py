# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import random

# Generating x values
x = [[1]*101, [], [], [], []]  # Initialize x with 101 ones and four empty lists for powers of x
x1 = []  # List to store x values
y = []   # List to store y values

# Generating x values from -5 to 5 in steps of 0.1 and corresponding y values
for i in range(-50, 51):
    i = i / 10  # Convert to float
    x1.append(i)  # Append x value to x1 list
    # Append powers of x to corresponding lists in x
    for j in range(1, 5):
        x[j].append(i ** j)
    # Generate y values with noise
    n = np.random.normal(0, 3)  # Generate random noise
    fx = (2 * (i ** 4)) - (3 * (i ** 3)) + (7 * (i ** 2)) - (23 * i) + 8 + n  # Polynomial function
    y.append(fx)  # Append y value to y list

# Transpose x and y for matrix operations
X = np.transpose(x)
Y = np.transpose(y)

# Calculate coefficients using normal equation for linear regression
b = np.dot(np.linalg.inv(np.dot(np.transpose(X), X)), np.dot(np.transpose(X), y))

# Predicting y values for different polynomial models
y1 = []  # Linear model
y2 = []  # Quadratic model
y3 = []  # Cubic model
y4 = []  # Biquadratic model
for i in x1:
    f1 = b[0] + b[1] * i
    y1.append(f1)
    f2 = b[0] + b[1] * i + b[2] * (i ** 2)
    y2.append(f2)
    f3 = b[0] + b[1] * i + b[2] * (i ** 2) + b[3] * (i ** 3)
    y3.append(f3)
    f4 = b[0] + b[1] * i + b[2] * (i ** 2) + b[3] * (i ** 3) + b[4] * (i ** 4)
    y4.append(f4)

# Function for Lagrange Interpolation
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

# Interpolating y values using Lagrange Interpolation
yInte = lagrangeInterpolation(x1, y, x1)

# Plotting the different models
plt.figure(figsize=(8, 4))
plt.plot(x1, y1, label="Linear")
plt.plot(x1, y2, label="Quadratic")
plt.plot(x1, y3, label="Cubic")
plt.plot(x1, y4, label="Biquadratic")
plt.plot(x1, yInte, marker='^', label="Lagrange Interpolation")
plt.xlabel('X')
plt.ylabel('Y = F(X)')
plt.title("Different Models and Their Plots")
plt.figtext(0.5, 0.01, "This plot shows the different models and their corresponding F(X) values.", ha="center",
            fontsize=10, bbox={"facecolor": "brown", "alpha": 0.5, "pad": 5})
plt.legend()
plt.show()

# Performing Bias-Variance Tradeoff analysis
X1 = random.sample(x1, 71)  # Randomly select 71 points for training
X2 = [i for i in x1 if i not in X1]  # Remaining points for testing

# Predicting y values for training points
y1_train = [b[0] + b[1] * i for i in X1]  # Linear model
y2_train = [b[0] + b[1] * i + b[2] * (i ** 2) for i in X1]  # Quadratic model
y3_train = [b[0] + b[1] * i + b[2] * (i ** 2) + b[3] * (i ** 3) for i in X1]  # Cubic model
y4_train = [b[0] + b[1] * i + b[2] * (i ** 2) + b[3] * (i ** 3) + b[4] * (i ** 4) for i in X1]  # Biquadratic model

# Calculating errors for training data
e1_train = [abs(y1_train[i] - y[i]) for i in range(71)]
e2_train = [abs(y2_train[i] - y[i]) for i in range(71)]
e3_train = [abs(y3_train[i] - y[i]) for i in range(71)]
e4_train = [abs(y4_train[i] - y[i]) for i in range(71)]
Etrain = [sum(e1_train) / len(e1_train), sum(e2_train) / len(e2_train), sum(e3_train) / len(e3_train),
          sum(e4_train) / len(e4_train)]

# Predicting y values for testing points
ytest1 = [b[0] + b[1] * i for i in X2]  # Linear model
ytest2 = [b[0] + b[1] * i + b[2] * (i ** 2) for i in X2]  # Quadratic model
ytest3 = [b[0] + b[1] * i + b[2] * (i ** 2) + b[3] * (i ** 3) for i in X2]  # Cubic model
ytest4 = [b[0] + b[1] * i + b[2] * (i ** 2) + b[3] * (i ** 3) + b[4] * (i ** 4) for i in X2]  # Biquadratic model

# Calculating errors for testing data
e1_test = [abs(ytest1[i] - y[i + 71]) for i in range(len(X2))]
e2_test = [abs(ytest2[i] - y[i + 71]) for i in range(len(X2))]
e3_test = [abs(ytest3[i] - y[i + 71]) for i in range(len(X2))]
e4_test = [abs(ytest4[i] - y[i + 71]) for i in range(len(X2))]
Etest = [sum(e1_test) / len(e1_test), sum(e2_test) / len(e2_test), sum(e3_test) / len(e3_test),
         sum(e4_test) / len(e4_test)]

# Plotting Bias-Variance Tradeoff
plt.figure(figsize=(8, 4))
plt.plot([1, 2, 3, 4], Etest, label="Variance")
plt.plot([1, 2, 3, 4], Etrain, label="Bias")
plt.xlabel('Complexity')
plt.ylabel('Error')
plt.title("Bias-Variance Tradeoff")
plt.figtext(0.5, 0.01, "This plot shows the Bias-Variance Tradeoff for different models.", ha="center", fontsize=10,
            bbox={"facecolor": "brown", "alpha": 0.5, "pad": 5})
plt.legend()
plt.show()
