
def highest_occuring_char(str):
    map = {}
    for c in str:
        if c.isalpha():
            c = c.lower()
            map[c] = map.get(c,0)+1
    
    max_char=""
    max_count=0
    for char, n in map.items():
        if n>max_count:
            max_char=char
            max_count=n
    
    return max_char,max_count


def main():
    print()


if __name__=="__main__":
    main()