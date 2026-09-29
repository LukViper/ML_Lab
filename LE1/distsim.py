def minkowski_dist(A,B,p):
    if len(A)!=len(B):
        raise ValueError("same length")
    total=0
    for i in range(len(A)):
        total+= abs(A[i]-B[i])**p
    return total**(1/p)

def dot_prod(A,B):
    if len(A)!=len(B):
        raise ValueError("same length")
    total=0
    for i in range(len(A)):
        total+=A[i]*B[i]
    return total

def euclidean_norm(A):
    total=0
    for i in range(len(A)):
        total+=A[i]**2
    return total**0.5

def cosine_sim(A,B):
    normA = euclidean_norm(A)
    normB = euclidean_norm(B)
    if normA == 0 or normB == 0:
        raise ValueError("zero")
    return dot_prod(A,B)/(normB*normA)


def jaccard_sim(A,B):
    if len(A)!=len(B):
        raise ValueError("same length")
    f11=0
    f10=0
    f01=0
    for i in range(len(A)):
        if (A[i]==1 and B[i]==1):
            f11+=1
        elif (A[i]==1 and B[i]==0):
            f10+=1
        elif (A[i]==0 and B[i]==1):
            f01+=1

    denom = f11+f01+f10
    if (denom==0):
        return 1.0
    return f11/denom

def SMC(A,B):
    if len(A)!=len(B):
        raise ValueError("same length")
    f11=0
    f00=0
    for i in range(len(A)):
        if (A[i]==1 and B[i]==1):
            f11+=1
        elif (A[i]==0 and B[i]==0):
            f00+=1
    return (f11+f00)/len(A)
        