import numpy as np

def drawSample(dict, n):
    cmf=[]
    randomNums=[]
    samples=[]
    keys=list(dict.keys())
    s=0
    for i in dict.values():
        s+=i
        cmf.append(s)
    for i in range(n):
        randomNums.append(np.random.uniform(0,1))
    for i in randomNums:
        j = 0
        while i > cmf[j]:
            j += 1
        samples.append(keys[j])
    return samples
        
dict = {'Apple': 0.5, 'Banana': 0.3, 'Carrot': 0.2}
n = 10
print(drawSample(dict, n))
