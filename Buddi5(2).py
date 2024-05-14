import numpy as np
import matplotlib.pyplot as plt
import random

x=[[1]*101,[],[],[],[]]
x1 =[]
y=[]
Y=[]

# Append x values to the respective columns in x matrix
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
x2=x1
np.random.shuffle(x2)
for i in x2:
    n=np.random.normal(0,3)
    fx=((2*(i**4))-(3*(i**3))+(7*(i**2))-(23*i)+8+n)
    y.append(fx)
# np.random.shuffle(x1)
# Transpose the 'x' list to prepare it for matrix operations
X=np.transpose(x)
# Transpose the 'y' list to prepare it for matrix operations
Y=np.transpose(y)
# Calculate the coefficients 'b' using linear regression formula
b=np.matmul(np.linalg.inv(np.matmul(np.transpose(X),X)),np.matmul(np.transpose(X),Y))
y1=[]
y2=[]
y3=[]
y4=[]
for i in x2:
    f1=b[0] + b[1]*i
    y1.append(f1)
    f2 = b[0] + b[1]*i + b[2] * (i**2)
    y2.append(f2)
    f3= b[0] + b[1]*i + b[2] * (i**2)  + b[3]*(i**3)
    y3.append(f3)
    f4= b[0] + b[1]*i + b[2] * (i**2)  + b[3]*(i**3) + b[4]*(i**4)
    y4.append(f4)

# print(len)
# for i in x1:
#     if i not in X1:
#         X2.append(i)
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
yInte=lagrangeInterpolation(x2,y,x2)
plt.figure(figsize=(8, 4))
plt.plot(x2,y1,label="linear")
plt.plot(x2,y2,label="quadratic")
plt.plot(x2,y3,label="cubic")
plt.plot(x2,y4,label="biquadratic")
plt.plot(x2,yInte,marker='^',label="legrange")
plt.xlabel('X')
plt.ylabel('Y=F(X)')
plt.title("Different models and its plot")
plt.figtext(0.5, 0.01, "Let us consider a biquadratic polynomial and 101 x values in the range(-5,5) and generated y values for different models such as linear,quadratic,cubic and biquadratic and their corresponding f(x) is plotted above", ha="center", fontsize=10, bbox={"facecolor":"brown", "alpha":0.5, "pad":5})
plt.legend()
#plt.show()

# x2=x1
# np.random.shuffle(x2)
X1=x1[:81]
X2=x1[81:]

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
# print(y1_train)
# print(y2_train)
# print(y3_train)
# print(y4_train)
e1train=[]
e2train=[]
e3train=[]
e4train=[]
com=[1,2,3,4]
for i in range(81):
    e1train.append((y1_train[i]-y[i])**2)
    e2train.append((y2_train[i]-y[i])**2)
    e3train.append((y3_train[i]-y[i])**2)
    e4train.append((y4_train[i]-y[i])**2)
Etrain=[]
Etrain.append(np.mean(e1train))
Etrain.append(np.mean(e2train))
Etrain.append(np.mean(e3train))
Etrain.append(np.mean(e4train))

ytest1=[]
ytest2=[]
ytest3=[]
ytest4=[]
Yorig=y[81:]
e1test=[]
e2test=[]
e3test=[]
e4test=[]
Etest=[]
for i in X2:
    f1=b[0] + b[1]*i
    ytest1.append(f1)
    f2 = b[0] + b[1]*i + b[2] * (i**2)
    ytest2.append(f2)
    f3= b[0] + b[1]*i + b[2] * (i**2)  + b[3]*(i**3)
    ytest3.append(f3)
    f4= b[0] + b[1]*i + b[2] * (i**2)  + b[3]*(i**3) + b[4]*(i**4)
    ytest4.append(f4)
print(len(ytest1))

# fxt=((2*(i**4))-(3*(i**3))+(7*(i**2))-(23*i)+8+n)
for i in range(len(ytest1)):
    e1test.append((ytest1[i]-Yorig[i])**2)
    e2test.append((ytest2[i]-Yorig[i])**2)
    e3test.append((ytest3[i]-Yorig[i])**2)
    e4test.append((ytest4[i]-Yorig[i])**2)
Etest.append(np.mean(e1test))
Etest.append(np.mean(e2test))
Etest.append(np.mean(e3test))
Etest.append(np.mean(e4test))
plt.figure(figsize=(8, 4))
plt.plot(com,Etest,label="Variance")
plt.plot(com,Etrain,label="Bias")
plt.xlabel('Complexity')
plt.ylabel('Error')
plt.title("Bias Variance TradeOff")
plt.figtext(0.5, 0.01, "The above graph shows the bias variance tradeoff for models such as linear,quadratic,cubic and biquadratic models", ha="center", fontsize=10, bbox={"facecolor":"brown", "alpha":0.5, "pad":5})
plt.legend()
plt.show()
