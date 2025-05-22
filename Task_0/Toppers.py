T = int(input())
for i in range(0, T):
    student_name = []
    score = []
    N = int(input())
    for j in range(0, N):
        sn, sc = [x for x in input().split(" ")]
        student_name.append(sn)
        score.append(float(sc))
    v = []
    for j in range(0, len(score)):
        if(score[j]==score[score.index(max(score))]):
            v.append(j)
    toppers = []
    for j in range(0, len(v)):
        toppers.append(student_name[v[j]])
    toppers.sort()
    for j in range(0, len(toppers)):
        print(toppers[j])
