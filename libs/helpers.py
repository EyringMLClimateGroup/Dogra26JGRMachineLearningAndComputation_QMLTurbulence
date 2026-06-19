import matplotlib.colors as mcolors
import numpy as np

def mix_with_white(color, amount=0.5):
    # color in matplotlib formats, output as rgb
    rgb = np.array(mcolors.to_rgb(color))
    white = np.array([1, 1, 1])
    mixed = (1 - amount) * rgb + amount * white
    return tuple(mixed)