from pyment.callbacks import Resettable, Resetter


def test_resetter_invalid_object():
    exception = False

    try:
        Resetter(None)
    except Exception:
        exception = True

    assert exception, ('Resetter does not throw an exception if object is not '
                       'Resettable')


def test_resetter_resets():
    class Test(Resettable):
        def __init__(self):
            self.call_count = 0

        def reset(self):
            self.call_count += 1

    obj = Test()
    resetter = Resetter(obj)
    resetter.on_epoch_end()

    assert 1 == obj.call_count, ('Resetter does not call obj.reset() on epoch '
                                 'end')