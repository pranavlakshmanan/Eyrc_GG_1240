def compute_distance(x1, y1, x2, y2):

    distance = 0.0
    distance = (((x2-x1) ** 2)+((y2-y1) ** 2))**0.5
    distance = format(distance, '.2f')
    print("Distance:", distance)


if __name__ == '__main__':

    test_cases = int(input())
    for i in range(0, test_cases):
        x1, y1, x2, y2 = [x for x in input().split(" ")]
        compute_distance(int(x1), int(y1), int(x2), int(y2))
    