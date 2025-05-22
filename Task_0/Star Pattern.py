T = int(input())
for l in range(0, T):
    N = int(input())
    i = N
    while (i > 0):
        j = i+i
        a = 0
        while (j > i):
            a += 1
            if (a != 5):
                print("*", end="")
            elif (a == 5):
                print("#", end="")
                a = 0
            j -= 1
        print("")
        i -= 1
