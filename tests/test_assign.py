"""The assignment problem: the Hungarian method gives the least total, checked against trying every assignment."""
import itertools
import random

import pytest

from robotnav.assign import hungarian, nearest_first, total


def _best_by_trying_all(cost):
    n, m = len(cost), len(cost[0])
    return min(total(cost, perm) for perm in itertools.permutations(range(m), n))


def test_a_small_case_by_hand():
    cost = [[4, 1, 3], [2, 0, 5], [3, 2, 2]]                # rows take columns 1, 0, 2: 1 + 2 + 2 = 5
    assert hungarian(cost) == [1, 0, 2] and total(cost, hungarian(cost)) == 5


@pytest.mark.parametrize("rows,cols", [(1, 1), (2, 2), (3, 3), (4, 4), (2, 4), (3, 5), (5, 5), (6, 6)])
def test_it_finds_the_least_total(rows, cols):
    rng = random.Random(rows * 31 + cols)
    for _ in range(60):
        cost = [[rng.uniform(1, 100) for _ in range(cols)] for _ in range(rows)]
        pick = hungarian(cost)
        assert len(set(pick)) == rows and all(0 <= j < cols for j in pick)     # every row its own column
        assert total(cost, pick) == pytest.approx(_best_by_trying_all(cost), rel=1e-12)


def test_ties_and_whole_numbers():
    rng = random.Random(7)
    for _ in range(200):
        cost = [[rng.randint(0, 5) for _ in range(4)] for _ in range(4)]
        assert total(cost, hungarian(cost)) == _best_by_trying_all(cost)


def test_a_cell_nobody_can_reach_is_avoided_if_it_can_be():
    big = 1e7
    cost = [[big, 3.0, 9.0], [2.0, big, big], [big, 8.0, 1.0]]
    assert hungarian(cost) == [1, 0, 2]


def test_nearest_first_is_never_better_and_sometimes_worse():
    rng = random.Random(3)
    worse = 0
    for _ in range(300):
        cost = [[rng.uniform(1, 100) for _ in range(4)] for _ in range(3)]
        best, greedy = total(cost, hungarian(cost)), total(cost, nearest_first(cost))
        assert best <= greedy + 1e-9
        worse += greedy > best + 1e-9
    assert worse > 30                                        # the obvious way loses often enough to matter
    cost = [[1.0, 2.0], [1.5, 50.0]]                         # the first row grabs what the second badly needs
    assert nearest_first(cost) == [0, 1] and hungarian(cost) == [1, 0]


def test_more_rows_than_columns_is_refused():
    with pytest.raises(ValueError):
        hungarian([[1.0], [2.0]])
