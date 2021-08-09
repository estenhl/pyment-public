from abc import abstractproperty, ABC
from tensorflow.keras.callbacks import Callback


class Resettable(ABC):
    @abstractproperty
    def reset(self) -> None:
        pass


class Resetter(Callback):
    def __init__(self, obj: Resettable):
        super().__init__()

        self.obj = obj

    def on_epoch_end(self, *args, **kwargs) -> None:
        self.obj.reset()