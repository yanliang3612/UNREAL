import math


def rbo_score(first_ranking, second_ranking, persistence):
    """Compute the rank-biased overlap used by UNREAL's node reordering."""
    if not first_ranking.any() or not second_ranking.any():
        return 0

    first_seen = set()
    second_seen = set()
    score = 0.0

    for depth in range(len(first_ranking)):
        first_seen.add(first_ranking[depth])
        second_seen.add(second_ranking[depth])
        average_overlap = len(first_seen & second_seen) / (depth + 1)
        score += math.pow(persistence, depth) * average_overlap

    return (1 - persistence) * score
