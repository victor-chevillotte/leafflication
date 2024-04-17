from matplotlib.backends.backend_agg import FigureCanvasBase
import numpy as np
import cv2


def plt_to_numpy_image(canvas: FigureCanvasBase) -> np.ndarray:
    canvas.draw()
    image_flat = np.frombuffer(canvas.tostring_rgb(), dtype="uint8")
    image = image_flat.reshape(*reversed(canvas.get_width_height()), 3)
    return image


def cv2_imshow_wrapper(window_name: str, image: np.ndarray):
    cv2.imshow(window_name, image)
    wait_time = 1000
    while (
        cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1
    ):
        keyCode = cv2.waitKey(wait_time)
        if (keyCode & 0xFF) == ord("q"):
            cv2.destroyAllWindows()
            break
