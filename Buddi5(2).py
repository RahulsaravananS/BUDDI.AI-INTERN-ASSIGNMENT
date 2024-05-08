import numpy as np
import matplotlib.pyplot as plt
x=[[1]*101,[],[],[],[]]
y=[]
for i in range(-50,51):
    i=i/10
    x[1].append(i)
    x[2].append(i**2)
    x[3].append(i**3)
    x[4].append(i**4)
    n=np.random.normal(0,3)
    Y=((2*(i**4))-(3*(i**3))+(7*(i**2))-(23*i)+8+n)
    y.append(Y)
x=np.array(x)
y=np.array(y)
x=np.transpose(x)
y=np.transpose(y)
b=np.dot(np.linalg.inv(np.dot(np.transpose(x),x)),np.dot(np.transpose(x),y))
print(b)