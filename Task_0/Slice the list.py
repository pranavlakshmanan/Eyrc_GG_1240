T = int(input())
for i in range(0, T):
    L = []
    length = int(input())
    L = [int(x) for x in input().split()]
    if (len(L) == length):
        L2 = L.copy()
        L2.reverse()
        for j in range(0, length):
            print(L2[j], end=" ")
        print()
        for j in range(0, length):
            if (j % 3 == 0 and j != 0):
                print(L[j]+3, end=" ")
        print()
        for j in range(0, length):
            if (j % 5 == 0 and j != 0):
                print(L[j]-7, end=" ")
        print()
        sum = 0
        for j in range(0, length):
            if (j >= 3 and j <= 7):
                sum += L[j]
        print(sum)
