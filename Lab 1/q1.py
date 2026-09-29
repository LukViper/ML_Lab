
def sum_of_pairs_equaling_ten(arr):
    ans=[]
    for i in range(len(arr)):
        for j in range(i+1,len(arr)):
            if i==j:
                continue
            if arr[i]+arr[j]==10:
                ans.append([arr[i],arr[j]])
    
    return ans


def main():
    ans = sum_of_pairs_equaling_ten([2,7,4,1,3,6])
    print(len(ans))

if __name__=="__main__":
    main()