import cv2

def capture_image(source):
    """
    Simulates camera capture.
    source can be:
    - image path
    - video frame snapshot
    """
    image = cv2.imread(source)
    if image is None:
        raise ValueError(f"Unable to load image from {source}")
    return image
