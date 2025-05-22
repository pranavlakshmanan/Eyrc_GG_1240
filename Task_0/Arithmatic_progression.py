from functools import reduce


def generate_AP(a1, d, n):

    AP_series = []
    for i in range(1, n+1):
        AP_series.append(a1+((i-1)*d))
    return AP_series


if __name__ == '__main__':
    test_cases = int(input())
    for i in range(0, test_cases):
        a1, d, n = [int(x) for x in input().split()]
        AP_series = generate_AP(a1, d, n)
        for i in range(0, len(AP_series)):
            print(AP_series[i], end=" ")
        print()
        sqr_AP_series = list(map(lambda x: x**2, AP_series))
        for i in range(0, len(sqr_AP_series)):
            print(sqr_AP_series[i], end=" ")
        print()
        sum_sqr_AP_series = reduce(lambda x, y: x+y, sqr_AP_series)
        print(sum_sqr_AP_series)