import math
def random_MMM():
    arr=[]
    sum=0
    mode=-1
    for i in range(25):
        arr[i]=math.random(1,10)
        sum+=arr[i]
        if mode<arr[i]:
            mode=arr[i]
        
    mean = sum/25
    sarr=sorted(arr)
    mid=len(sarr)//2
    
    
    