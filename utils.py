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
    while cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1:
        keyCode = cv2.waitKey(wait_time)
        if (keyCode & 0xFF) == ord("q"):
            cv2.destroyAllWindows()
            break


def pyqt_setup():
    import os

    VIRTUAL_ENV_PATH = os.environ.get("VIRTUAL_ENV")
    if VIRTUAL_ENV_PATH is None:
        print("Virtual environment not found.")
        print("Please activate the virtual environment.")
        exit(1)
    QT_PLUGIN_PATH = os.path.join(
        VIRTUAL_ENV_PATH,
        "lib/python3.10/site-packages/cv2/qt/plugins/platforms",
    )
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QT_PLUGIN_PATH
