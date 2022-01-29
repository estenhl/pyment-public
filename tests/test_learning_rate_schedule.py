import os

from pyment.utils.learning_rate import LearningRateSchedule

from utils import assert_exception


def test_learning_rate_schedule():
    schedule = LearningRateSchedule({0: 1e-2, 5: 1e-3, 10: 1e-4})

    learning_rates = [schedule(i) for i in range(20)]

    expected = [1e-2] * 5 + [1e-3] * 5 + [1e-4] * 10

    assert expected == learning_rates, \
        'LearningRateSchedule does not produce the correct learning rates'

def test_learning_rate_no_initial():
    assert_exception(LearningRateSchedule, args=[{1: 1}], exception=ValueError, 
                      message=('Instantiating a LearningRateSchedule without '
                               'a learning rate for epoch 0 does not raise '
                               'an error'))

def test_learning_rate_negative_epoch():
    schedule = LearningRateSchedule({0: 1e-2, 5: 1e-3, 10: 1e-4})

    assert_exception(schedule, args=[-1], exception=ValueError,
                     message=('Calling LearningRateSchedule with epoch < 0 '
                              'does not raise an error'))

def test_learning_rate_to_from_json():
    schedule = LearningRateSchedule({0: 1e-2, 5: 1e-3, 10: 1e-4})
    schedule = LearningRateSchedule.from_json(schedule.json)

    learning_rates = [schedule(i) for i in range(20)]

    expected = [1e-2] * 5 + [1e-3] * 5 + [1e-4] * 10

    assert expected == learning_rates, \
        ('LearningRateSchedule to and from json does not produce the correct '
         'learning rates')

def test_learning_rate_to_from_jsonstring():
    schedule = LearningRateSchedule({0: 1e-2, 5: 1e-3, 10: 1e-4})
    schedule = LearningRateSchedule.from_jsonstring(schedule.jsonstring)

    learning_rates = [schedule(i) for i in range(20)]

    expected = [1e-2] * 5 + [1e-3] * 5 + [1e-4] * 10

    assert expected == learning_rates, \
        'LearningRateSchedule does not produce the correct learning rates'

def test_learning_rate_from_json_without_schedule():
    assert_exception(LearningRateSchedule.from_json, 
                     args=[{'not_schedule': {}}], exception=KeyError,
                     message=('Calling LearningRateSchedule.from_json with '
                              'an object without a \'schedule\' field does '
                              'not raise an error'))

def test_learning_rate_schedule_save_load():
    try:
        schedule = LearningRateSchedule({0: 1e-2, 5: 1e-3, 10: 1e-4})

        before = [schedule(i) for i in range(20)]

        schedule.save('tmp.json')
        schedule = LearningRateSchedule.from_jsonfile('tmp.json')

        after = [schedule(i) for i in range(20)]

        expected = [1e-2] * 5 + [1e-3] * 5 + [1e-4] * 10

        assert expected == before == after, \
            ('LearningRateSchedule save and load does not produce the ' \
             'correct learning rates')
    finally:
        if os.path.isfile('tmp.json'):
            os.remove('tmp.json')

