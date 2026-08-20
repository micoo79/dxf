"""Unicode-biztos képbeolvasás.

Windowson a cv2.imread nem boldogul az ékezetes útvonalakkal (pl.
C:\\Users\\...\\Képek\\...), ezért ilyenkor bájtokból dekódolunk.
"""

import numpy as np
import cv2


def imread(path, flags=cv2.IMREAD_COLOR):
    """cv2.imread, amely ékezetes/unicode útvonalakon is működik."""
    img = cv2.imread(path, flags)
    if img is None:
        try:
            data = np.fromfile(path, dtype=np.uint8)
            if data.size:
                img = cv2.imdecode(data, flags)
        except OSError:
            img = None
    return img
