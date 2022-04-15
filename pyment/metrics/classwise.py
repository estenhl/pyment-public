import tensorflow as tf

from tensorflow.keras.metrics import AUC, Precision, Recall

from .utils import binarize


class ClasswisePrecision(Precision):
    def __init__(self, index: int, name: str = None, epsilon: float = 1e-9):
        if name is None:
            name = f'classwise_precision_{index}'

        super().__init__(name=name)

        self.index = index

    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor,
                     **kwargs) -> tf.Tensor:
        y_true = binarize(y_true, index=self.index)
        y_pred = binarize(y_pred, index=self.index)

        return super().update_state(y_true, y_pred, **kwargs)


class ClasswiseRecall(Recall):
    def __init__(self, index: int, name: str = None, epsilon: float = 1e-9):
        if name is None:
            name = f'classwise_recall_{index}'

        super().__init__(name=name)

        self.index = index

    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor,
                     **kwargs) -> tf.Tensor:
        y_true = binarize(y_true, index=self.index)
        y_pred = binarize(y_pred, index=self.index)

        return super().update_state(y_true, y_pred, **kwargs)


class ClasswiseAUC(AUC):
    def __init__(self, index: int, name: str = None, epsilon: float = 1e-9):
        if name is None:
            name = f'classwise_auc_{index}'

        super().__init__(name=name)

        self.index = index

    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor,
                     **kwargs) -> tf.Tensor:
        y_true = binarize(y_true, index=self.index)
        y_pred = binarize(y_pred, index=self.index)

        return super().update_state(y_true, y_pred, **kwargs)
