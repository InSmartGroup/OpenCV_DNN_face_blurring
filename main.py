import cv2
import numpy as np


def pixelate(roi, pixels=16):

    # Size of region to pixelate
    roi_h, roi_w = roi.shape[:2]

    if roi_h > pixels and roi_w > pixels:

        # Resize input ROI to the (small) pixelated size
        roi_small = cv2.resize(roi, (pixels, pixels), interpolation=cv2.INTER_LINEAR)

        # Enlarge the pixelated ROI to fill the size of the original ROI
        roi_pixelated = cv2.resize(roi_small, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)

    else:
        roi_pixelated = roi

    return roi_pixelated


def face_blur_pixelate(image, model, detection_threshold=0.9, pixels=10):
    image = image.copy()

    # Prepare image and perform inference.
    blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(300, 300), mean=[104, 117, 123])
    model.setInput(blob)
    detections = model.forward()

    h, w = image.shape[:2]

    for i in range(detections.shape[2]):

        confidence = detections[0, 0, i, 2]

        if confidence > detection_threshold:

            # Extract the bounding box coordinates from the detection
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")

            face = image[y1:y2, x1:x2]
            face = pixelate(face, pixels=pixels)
            image[y1:y2, x1:x2] = face

    return image


# Caffe model parameters
model_file = 'caffe_model/res10_300x300_ssd_iter_140000.caffemodel'
config_file = 'caffe_model/deploy.prototxt'

# Read the model and create a network object.
model = cv2.dnn.readNetFromCaffe(prototxt=config_file, caffeModel=model_file)

# Video capture object
video_cap = cv2.VideoCapture(0)

# Window params
win_name = 'Face blurring'


# Process the web camera stream
while True:
    has_frame, frame = video_cap.read()
    if not has_frame:
        break

    frame = cv2.flip(frame, 1)

    frame = face_blur_pixelate(frame, model)

    cv2.imshow(win_name, frame)

    # Keyboard control
    key = cv2.waitKey(1)

    if key == 27 or key == ord('q') or key == ord('Q'):
        break

video_cap.release()
cv2.destroyWindow(win_name)
