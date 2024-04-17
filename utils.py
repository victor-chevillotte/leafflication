from matplotlib.backends.backend_agg import FigureCanvasBase
import numpy as np


def plt_to_numpy_image(canvas: FigureCanvasBase) -> np.ndarray:
    canvas.draw()
    image_flat = np.frombuffer(canvas.tostring_rgb(), dtype="uint8")
    image = image_flat.reshape(*reversed(canvas.get_width_height()), 3)
    return image
