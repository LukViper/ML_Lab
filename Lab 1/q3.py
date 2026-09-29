
def matA_power_M(arr,m):
    rows = len(arr)
    cols = len(arr[0])
    if rows != cols:
        return "not a square matrix"
    res = arr
    for _ in range(m-1):
        res=multiple_matrices(res,arr)
    return res

    

def multiple_matrices(mat1,mat2):
    n = len(mat1)
    ans = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                ans[i][j]+=mat1[i][k]*mat2[k][j]
    return ans


def main():
    print(matA_power_M([[2,3],[3,4]],3))

if __name__=="__main__":
    main()