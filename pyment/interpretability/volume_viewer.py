import matplotlib.pyplot as plt
import numpy as np

from ipywidgets import interact, IntSlider
from typing import Tuple


class VolumeViewer:
    def __init__(self, image: np.ndarray, figsize: Tuple[int, int] = (5, 5), 
                 title: str = None):
        self.image = image
        
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.img = None

        self.ax.axis('off')

        if title is not None:
            self.fig.suptitle(title)
        
        interact(self.plot_slice, 
                 layer=IntSlider(min=0, max=len(self.image)-1, step=1, 
                                 value=10, continuous=False))
        
    def plot_slice(self, layer: int):
        if self.img is None:
            self.img = self.ax.imshow(self.image[layer])
        else:
            self.img.set_data(self.image[layer])