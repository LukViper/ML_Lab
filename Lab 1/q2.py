
def range_of_list(arr):
    if len(arr)<=3:
        return -1
    return find_max(arr)-find_min(arr)

def find_max(arr):
    max=-1
    for i in range(arr):
        if arr[i]>max:
            max=arr[i]
    return max

def find_min(arr):
    min=1000000000000
    for i in range(arr):
        if arr[i]<min:
            min=arr[i]
    return min

def main():
    print(range_of_list([1,2,3,4,5,6,6,7,8,9]))

if __name__=="__main__":
    main()