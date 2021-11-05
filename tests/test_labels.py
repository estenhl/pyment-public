from pyment.labels import BinaryLabel, ContinuousLabel, Label


def test_get_binary_label():
    label = Label.from_type('binary', name='test')

    assert isinstance(label, BinaryLabel), \
        'Label.from_type(binary) does not return a BinaryLabel'

def test_get_continuous_label():
    label = Label.from_type('continuous', name='test')

    assert isinstance(label, ContinuousLabel), \
        'Label.from_type(continuous) does not return a ContinuousLabel'

def test_get_label_kwargs():
    label = Label.from_type('binary', name='test')

    assert 'test' == label.name, \
        'Label.from_type does not utilize additional keyword arguments'