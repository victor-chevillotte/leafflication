import matplotlib.pyplot as plt
import numpy as np


def plt_to_numpy_image(plot: plt) -> np.ndarray:
    fig = plot.figure(figsize=(10, 6))
    canvas = fig.canvas
    canvas.draw()
    image_flat = np.frombuffer(canvas.tostring_rgb(), dtype="uint8")
    image = image_flat.reshape(*reversed(canvas.get_width_height()), 3)
    return image
