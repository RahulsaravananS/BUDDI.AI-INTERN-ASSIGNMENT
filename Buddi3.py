import numpy as np

def drawSample(dict, n):
    pmf=[]
    cmf=[]
    randomNums=[]
    samples=[]
    keys=list(dict.keys()) 
    sum_keys=sum(dict.values())
    for i in dict.values():
        pmf.append(i/sum_keys)
        cmf.append(sum(pmf))
    for i in range(n):
        randomNums.append(np.random.uniform(0,1))
    for i in randomNums:
        j = 0
        while i > cmf[j]:
            j += 1
        samples.append(keys[j])
    return samples
        
dict = {'Apple': 10, 'Banana': 6, 'Carrot': 4}
n = 10
print(drawSample(dict, n))
