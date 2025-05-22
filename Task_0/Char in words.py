T = int(input())
for i in range(0, T):
    str = input()
    count1 = 0
    count2 = 0
    for count1 in range(len(str)-1,0,-1):
        if (str[count1] != " "):
            count2 += 1
        elif (str[count1] == " "):
            print(count2,"%c"%',',end="")
            count2 = 0

    print("")
