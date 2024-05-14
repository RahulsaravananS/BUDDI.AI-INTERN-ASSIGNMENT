# Import necessary modules or libraries

import numpy as np
import matplotlib.pyplot as plt

# Initialize the lists

x=[[1]*101,[],[],[],[]]
x1 =[]
Y=[]

# Generate x values from -5 to 5 with a step size of 0.1 and calculate y values

for i in range(-50,51):
    i=i/10
    x1.append(i)
    x[1].append(i)
    x[2].append(i**2)
    x[3].append(i**3)
    x[4].append(i**4)
    n=np.random.normal(0,3)
    fx=((2*(i**4))-(3*(i**3))+(7*(i**2))-(23*i)+8+n)
    Y.append(fx)

# Transpose the 'X' and 'Y' list to prepare it for matrix operations

X=np.transpose(x)
Y=np.transpose(Y)

# Calculate the coefficients 'b' using linear regression formula

b=np.matmul(np.linalg.inv(np.matmul(np.transpose(X),X)),np.matmul(np.transpose(X),Y))

# Initialize list for store the output of different models

y1=[]
y2=[]
y3=[]
y4=[]

# Calculating Values

for i in x1:
    f1=b[0] + b[1]*i # Linear
    y1.append(f1)
    f2 = b[0] + b[1]*i + b[2] * (i**2) # Quadratic
    y2.append(f2)
    f3= b[0] + b[1]*i + b[2] * (i**2)  + b[3]*(i**3) # Cubic
    y3.append(f3)
    f4= b[0] + b[1]*i + b[2] * (i**2)  + b[3]*(i**3) + b[4]*(i**4) # Biquadratic
    y4.append(f4)

# Function that defines lagrange interpolation

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
yInte=lagrangeInterpolation(x1,Y,x1)

# Plot the different models

plt.figure(figsize=(8, 4))
plt.plot(x1,y1,label="linear")
plt.plot(x1,y2,label="quadratic")
plt.plot(x1,y3,label="cubic")
plt.plot(x1,y4,label="biquadratic")
plt.plot(x1,yInte,marker='^',label="legrange")
plt.xlabel('X')
plt.ylabel('Y=F(X)')
plt.title("Different models and its plot")
plt.figtext(0.5, 0.01, "Let us consider a biquadratic polynomial and 101 x values in the range(-5,5) and generated y values for different models such as linear,quadratic,cubic and biquadratic and their corresponding f(x) is plotted above", ha="center", fontsize=10, bbox={"facecolor":"brown", "alpha":0.5, "pad":5})
plt.legend()

# shuffle the data and get 80% training and 20% testing Data

x2=x1
np.random.shuffle(x2)

# Training input
X1=x1[:81]
# Testing input
X2=x1[81:]
#Actual Training output
y1=[]
#Actual Testing output
y2=[]
for i in X1:
    n=np.random.normal(0,3)
    fx=((2*(i**4))-(3*(i**3))+(7*(i**2))-(23*i)+8+n)
    y1.append(fx)
for i in X2:
    n=np.random.normal(0,3)
    fx=((2*(i**4))-(3*(i**3))+(7*(i**2))-(23*i)+8+n)
    y2.append(fx)

# Calculate Predicted Training data Output for Different model

y1_train=[]
y2_train=[]
y3_train=[]
y4_train=[]
for i in X1:
    f1=b[0] + b[1]*i
    y1_train.append(f1)
    f2 = b[0] + b[1]*i + b[2] * (i**2)
    y2_train.append(f2)
    f3= b[0] + b[1]*i + b[2] * (i**2)  + b[3]*(i**3)
    y3_train.append(f3)
    f4= b[0] + b[1]*i + b[2] * (i**2)  + b[3]*(i**3) + b[4]*(i**4)
    y4_train.append(f4)

# Calculate Training error for different model

e1train=[]
e2train=[]
e3train=[]
e4train=[]

# Define model complexity
com=[1,2,3,4]

#Calculating MSE for Training data
for i in range(81):
    e1train.append((y1_train[i]-y1[i])**2)
    e2train.append((y2_train[i]-y1[i])**2)
    e3train.append((y3_train[i]-y1[i])**2)
    e4train.append((y4_train[i]-y1[i])**2)

# Store average Training error for particular model and store in list
Etrain=[]
Etrain.append(np.mean(e1train))
Etrain.append(np.mean(e2train))
Etrain.append(np.mean(e3train))
Etrain.append(np.mean(e4train))

# declare list to store Testing data output for different model
ytest1=[]
ytest2=[]
ytest3=[]
ytest4=[]

# Calculate Testing error for different model
e1test=[]
e2test=[]
e3test=[]
e4test=[]
Etest=[]

# Calculate Predicted Testing data Output for Different model
for i in X2:
    f1=b[0] + b[1]*i
    ytest1.append(f1)
    f2 = b[0] + b[1]*i + b[2] * (i**2)
    ytest2.append(f2)
    f3= b[0] + b[1]*i + b[2] * (i**2)  + b[3]*(i**3)
    ytest3.append(f3)
    f4= b[0] + b[1]*i + b[2] * (i**2)  + b[3]*(i**3) + b[4]*(i**4)
    ytest4.append(f4)

#Calculating MSE for Testing data
for i in range(len(ytest1)):
    e1test.append((ytest1[i]-y2[i])**2)
    e2test.append((ytest2[i]-y2[i])**2)
    e3test.append((ytest3[i]-y2[i])**2)
    e4test.append((ytest4[i]-y2[i])**2)

# Store average Testing error for particular model and store in list
Etest.append(np.mean(e1test))
Etest.append(np.mean(e2test))
Etest.append(np.mean(e3test))
Etest.append(np.mean(e4test))

# plot the bias-variance trade-off graph
plt.figure(figsize=(8, 4))
plt.plot(com,Etest,label="Variance")
plt.plot(com,Etrain,label="Bias")
plt.xlabel('Complexity')
plt.ylabel('Error')
plt.title("Bias Variance TradeOff")
plt.figtext(0.5, 0.01, "The above graph shows the bias variance tradeoff for models such as linear,quadratic,cubic and biquadratic models", ha="center", fontsize=10, bbox={"facecolor":"brown", "alpha":0.5, "pad":5})
plt.legend()
plt.show()
