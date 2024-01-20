import math

def rbo_score(l1, l2, p):
    if not l1.any() or  not l2.any():
        return 0
    s1 = set()
    s2 = set()
    max_depth = len(l1)
    score = 0.0
    for d in range(max_depth):
        s1.add(l1[d])
        s2.add(l2[d])
        avg_overlap = len(s1 & s2) / (d + 1)
        score += math.pow(p, d) * avg_overlap
    return (1 - p) * score
