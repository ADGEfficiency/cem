import numpy as np

from cem import get_elite_indicies


def test_elite_selection():

    rewards = np.array([5, 7, -2, 11])
    num_elite = 3

    elite_indicies = get_elite_indicies(num_elite, rewards)

    assert elite_indicies == [3, 1, 0]
