import numpy as np
a=[10,6,4]
b=['Banana','Apple','Carrot']
def pmf(a):
    c=[]
    d=sum(a)
    for i in a:
        c.append(i/d)
    return c
def cmf(e):
    m=[]
    for i in range(len(e)):
        m.append(sum(e[::i+1]))
    return m
e=pmf(a)
print("The pmf is",e)
f=cmf(e)
f=f[::-1]
print("The cmf is",f)
print("Enter the number of sample you want")
n=int(input())
z=[]
for i in range(n):
    x=np.random.uniform(0,1)
    for j in range(len(e)):
        if f[j] > x:
            z.append(j)
            break
print("The generated random sample index is",z)
print("The generated random sample is",end=" ")
for i in z:
    print(b[i],end=" ")
print()