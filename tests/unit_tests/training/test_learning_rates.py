import math
import mock
import pytest
from pytest_mock import mocker

import torch.nn as nn
import torch.optim as optim
from training import learning_rates

# Instructions
# https://medium.com/@bfortuner/python-unit-testing-with-pytest-and-mock-197499c4623c


## Shared Objects

INITIAL_LR = 1e-3

@pytest.fixture(scope="module")
def example_fixture():
    return 1e-3

@pytest.fixture(scope="module")
def lr_schedule():
    return {
        1: 1e-3,
        5: 1e-4,
        10: 1e-5
    }

def sgd():
    model = nn.Sequential(nn.Linear(3, 3))
    return optim.SGD(model.parameters(), lr=INITIAL_LR)

def adam():
    model = nn.Sequential(nn.Linear(3, 3))
    return optim.Adam(model.parameters(), lr=INITIAL_LR)


## Tests

def test_get_learning_rate():
    LR = learning_rates.LearningRate(INITIAL_LR, 'epoch')
    optim = sgd()
    assert LR.initial_lr == INITIAL_LR
    assert LR.get_learning_rate(optim) == INITIAL_LR

def test_set_learning_rate():
    LR = learning_rates.LearningRate(INITIAL_LR, 'epoch')
    optim = sgd()
    new_lr = INITIAL_LR + 1e-1
    LR.set_learning_rate(optim, new_lr)
    assert LR.get_learning_rate(optim) == new_lr

def test_LearningRate_adjust():
    LR = learning_rates.LearningRate(INITIAL_LR, 'epoch')
    optim = sgd()
    iteration = 5
    new_lr_expected = INITIAL_LR + 1e-1
    new_lr_output = LR.adjust(optim, new_lr_expected, iteration)
    assert new_lr_output == new_lr_expected
    assert LR.get_learning_rate(optim) == new_lr_expected
    assert LR.lr_history[0] == [iteration, new_lr_expected]

def test_FixedLR_adjust():
    LR = learning_rates.FixedLR(INITIAL_LR, 'epoch')
    optim = sgd()
    iteration = 5
    new_lr_output = LR.adjust(optim, iteration)
    assert new_lr_output == INITIAL_LR
    assert LR.get_learning_rate(optim) == INITIAL_LR
    assert LR.lr_history[0] == [iteration, INITIAL_LR]

def test_LinearLR_adjust():
    fixed_delta = 1e-1
    LR = learning_rates.LinearLR(INITIAL_LR, 'epoch', fixed_delta)
    optim = sgd()
    iteration = 5
    new_lr_expected = INITIAL_LR + fixed_delta
    new_lr_output = LR.adjust(optim, iteration)
    assert new_lr_output == new_lr_expected
    assert LR.get_learning_rate(optim) == new_lr_expected
    assert LR.lr_history[0] == [iteration, new_lr_expected]

def test_ScheduledLR_adjust(lr_schedule):
    LR = learning_rates.ScheduledLR(INITIAL_LR, 'epoch', lr_schedule)
    optim = sgd()

    assert LR.adjust(optim, 1) == lr_schedule[1]
    assert LR.get_learning_rate(optim) == lr_schedule[1]
    assert LR.lr_history[0] == [1, lr_schedule[1]]

    assert LR.adjust(optim, 2) == lr_schedule[1]
    assert LR.get_learning_rate(optim) == lr_schedule[1]
    assert LR.lr_history[1] == [2, lr_schedule[1]]

    assert LR.adjust(optim, 5) == lr_schedule[5]
    assert LR.get_learning_rate(optim) == lr_schedule[5]
    assert LR.lr_history[2] == [5, lr_schedule[5]]

def test_SnapshotLR_adjust():
    max_lr = 0.1
    n_iters = 100
    n_cycles = 5
    LR = learning_rates.SnapshotLR(INITIAL_LR, 'mini_batch',
                                    max_lr, n_iters, n_cycles)
    optim = sgd()

    assert LR.adjust(optim, 0) == 0.1
    assert LR.get_learning_rate(optim) == 0.1
    assert LR.lr_history[0] == [0, max_lr]

    assert math.isclose(LR.adjust(optim, 19), 0.0006, abs_tol=0.000016)
    assert math.isclose(LR.get_learning_rate(optim), 0.0006, abs_tol=0.000016)

    assert LR.adjust(optim, 20) == 0.1
    assert LR.get_learning_rate(optim) == 0.1

