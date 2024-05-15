# Import necessary modules or libraries
import numpy as np
import matplotlib.pyplot as plt

# Function to convert 'B' and 'G' labels to indicator variables (1 for 'B', 0 for 'G')
def indicator(Y):
    ind=[]
    for i in Y:
        if i=="B":
            ind.append(1)
        else:
            ind.append(0)
    return ind

# Function to classify a value based on a threshold
def discriminant(x, threshold):
    if x < threshold:
        return "G"
    else:
        return "B"
    
# Function to fit a linear regression model
def fit_linear_regression(a, Y):
    # Adding constant term for matrix manipulation
    X=[[1]*len(a)]
    X.append(a)
    X=np.array(X)
    X=np.transpose(X)
    
    # Computing betas using closed-form solution
    betas=np.matmul(np.linalg.inv(np.matmul(np.transpose(X),X)),np.matmul(np.transpose(X),Y))
    b0=betas[0]
    b1=betas[1]
    return b1, b0

# Function to calculate the value of y for a given x using the regression line equation
def calculate_value(x_new):
    return slope * x_new + intercept

# Input data
X = np.array([1,2,3,4,5,-3,-4,-5,-6])
Y = np.array(['B','B','B','B','B','G','G','G','G'])

# Convert labels to indicator variables
Indicator = indicator(Y)

# Calculate threshold for classification
threshold = np.mean(Indicator)

# Convert labels to indicator variables
Indicator = indicator(Y)

# Fit linear regression model
slope, intercept = fit_linear_regression(X, Indicator)

# Calculate value of y for a new data point
x_new = 8
regressValue = calculate_value(x_new)

# Classify the new data point
clas = discriminant(regressValue, threshold)

# Print classification result
print("The class of given data point", x_new, "is", clas)
print("Slope (b1):", slope)
print("Intercept (b0):", intercept)

# Calculate the midpoint of the regression line
midpoint_x = np.mean(X)
midpoint_y = calculate_value(midpoint_x)

# Define slope of perpendicular line (negative reciprocal of the slope of regression line)
perpendicular_slope = -1 / slope

# Define equation of perpendicular line: y - y1 = m * (x - x1)
# Where (x1, y1) is the midpoint of the regression line
def perpendicular_line_equation(x):
    return perpendicular_slope * (x - midpoint_x) + midpoint_y

# Plotting
plt.scatter(X[:5], Indicator[:5], color='blue', marker="*", label='Class A Data points')
plt.scatter(X[5:], Indicator[5:], color='orange', marker="o", label='Class B Data points')

# Plot regression line
plt.plot(X, calculate_value(X), color='red', label='Regression line')

# Plot perpendicular line
plt.plot(X, perpendicular_line_equation(X), linestyle='--', color='green', label='Decision Boundary')

plt.xlabel('X')
plt.ylabel('Y')
plt.ylim(-0.5,1.5)
plt.title('Linear Regression with Decision Boundary')
plt.figtext(0.5, 0.01, "In the above graph, the decision boundary is perpendicular to the regression line which classify 2 classes", ha="center", fontsize=10, bbox={"facecolor":"brown", "alpha":0.5, "pad":5})
plt.legend()
plt.show()
