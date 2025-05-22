bin_num = ""
count = 0


def dec_to_binary(n):
    global count, bin_num
    count += 1
    if (n > 1 or count < 8):
        dec_to_binary(n//2)
    bin_num = bin_num + str(n % 2)
    return bin_num


if __name__ == '__main__':
    T = int(input())
for j in range(0, T):
    n = int(input())
    bin_num = dec_to_binary(n)
    print(bin_num)
    bin_num = ""
    count=0
