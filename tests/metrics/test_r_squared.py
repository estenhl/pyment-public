import numpy as np

from pyment.metrics.r_squared import R2


def test_result_is_one_for_perfect_predictions():
    metric = R2()
    metric.update_state([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0])

    assert np.isclose(metric.result().numpy(), 1.0), (
        'Expected R2 to return 1.0 for perfect predictions'
    )


def test_result_is_zero_for_mean_predictions():
    metric = R2()
    metric.update_state([1.0, 2.0, 3.0, 4.0], [2.5, 2.5, 2.5, 2.5])

    assert np.isclose(metric.result().numpy(), 0.0), (
        'Expected R2 to return 0.0 for constant predictions at the mean'
    )


def test_result_is_consistent_across_batches():
    single = R2()
    single.update_state([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0])

    batched = R2()
    batched.update_state([1.0, 2.0], [1.0, 2.0])
    batched.update_state([3.0, 4.0], [3.0, 4.0])

    assert np.isclose(single.result().numpy(), batched.result().numpy()), (
        'Expected R2 to return the same result for batched and single updates'
    )


def test_reset_state_zeros_all_accumulators():
    metric = R2()
    metric.update_state([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    metric.reset_state()

    assert all(v.numpy() == 0.0 for v in metric.variables), (
        'Expected reset_state to zero all accumulators'
    )
