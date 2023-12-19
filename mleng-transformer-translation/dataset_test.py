import numpy as np

from dataset import getitem


def test_getitem_src():
    idx = 1
    data = np.array(
        [
            [4, 5, 6, 7, 0],
            [8, 9, 0, 0, 0],
            [10, 11, 12, 13, 14],
            [15, 0, 0, 0, 0],
            [16, 17, 18, 0, 0],
        ]
    )
    data_lengths = np.array([4, 2, 5, 1, 3])
    batches = [np.array([2]), np.array([0, 1, 3]), np.array([4])]
    src = True

    batch, masks = getitem(idx, data, data_lengths, batches, src)
    assert np.array_equal(
        batch, np.array([[4, 5, 6, 7], [8, 9, 0, 0], [15, 0, 0, 0]])
    )

    assert np.array_equal(
        masks,
        np.array(
            [
                [False, False, False, False],
                [False, False, True, True],
                [False, True, True, True],
            ]
        ),
    )


def test_getitem_tgt():
    idx = 1
    data = np.array(
        [
            [4, 5, 6, 7, 0],
            [8, 9, 0, 0, 0],
            [10, 11, 12, 13, 14],
            [15, 0, 0, 0, 0],
            [16, 17, 18, 0, 0],
        ]
    )
    data_lengths = np.array([4, 2, 5, 1, 3])
    batches = [np.array([2]), np.array([0, 1, 3]), np.array([4])]
    src = False

    batch, masks = getitem(idx, data, data_lengths, batches, src)
    assert np.array_equal(
        batch, np.array([[2, 4, 5, 6, 7, 3], [2, 8, 9, 3, 0, 0], [2, 15, 3, 0, 0, 0]])
    )

    assert np.array_equal(
        masks,
        np.array(
            [
                [False, False, False, False, False, False],
                [False, False, False, False, True, True],
                [False, False, False, True, True, True],
            ]
        ),
    )
