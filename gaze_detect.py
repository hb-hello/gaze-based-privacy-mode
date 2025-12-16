from gaze_tracking import GazeTracking
import cv2
import logging
import os
import time
from scipy.spatial import distance
import concurrent.futures

# Single-threaded executor for asynchronous file writes
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

MODELS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
FRAMES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'annotated_frames')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def detect(gray, frame):
    all_roi_faces=[]
    all_coordinates=[]
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for i,(x, y, w, h) in enumerate(faces):
        roi_color = frame[y:y+h, x:x+w]
        all_roi_faces.append(roi_color)
        all_coordinates.append([x+(w/2), y+(h/2)])
    return all_roi_faces,all_coordinates

def are_there_multiple_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) > 1:
        return True
    else:
        return False

def is_there_one_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    print(len(faces))
    print(faces)
    if len(faces) == 1:
        return True
    else:
        return False

def getGazeAttention(image, frame_counter):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #converts frames to gray scale
    image_x = gray.shape[0]
    image_y = gray.shape[1]

    x_mid = image_x/2
    y_mid = image_y/2

    del_x=0
    del_y=0

    threshold = 0.16

    gaze = GazeTracking()
    print("Gaze Tracking")

    gaze.refresh(image)
    text=""

    t1=0
    t2=0
    R=0
    C=0
    L=0

    # attention_percent = ((len(all_faces) - total_del)*100)/(len(all_faces)+0.1)
    print('Pupils?' + str(gaze.pupils_located))
    text = 'Could not detect pupils.'

    if(gaze.pupils_located):
        text = f'{gaze.eye_left.center[0]}, {gaze.eye_left.center[1]} : {gaze.eye_left.pupil.x}, {gaze.eye_left.pupil.y}'
        pupil_left = [gaze.eye_left.pupil.x, gaze.eye_left.pupil.y]
        pupil_right = [gaze.eye_right.pupil.x, gaze.eye_right.pupil.y]
        metric_left = distance.euclidean(gaze.eye_left.center, pupil_left) / (((gaze.eye_left.center[0] ** 2)+(gaze.eye_left.center[0] ** 2)) ** 0.5)
        metric_right = distance.euclidean(gaze.eye_right.center, pupil_right) / (((gaze.eye_right.center[0] ** 2)+(gaze.eye_right.center[0] ** 2)) ** 0.5)
        metric = (metric_left + metric_right) / 2
        if metric < threshold:
            attention_percent = 1.0 - metric
        else:
            attention_percent = max(0.6 - metric, 0.0)
    else:
        attention_percent = 0.0

    attention_percent = attention_percent * 100

    formatted_time = time.strftime('%b%d-%H-%M', time.localtime(time.time()))
    image_file = os.path.join(FRAMES_PATH, f'frame_{formatted_time}_{frame_counter}.png')
    annotated_frame = cv2.putText(gaze.annotated_frame(), f'Attention: {str(attention_percent)}', (10,15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,0), 1)
    annotated_frame = cv2.putText(annotated_frame, f'{text}', (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,0), 1)
    # annotated_frame = cv2.putText(annotated_frame, f'{metric_left}, {metric_right}', (10,120), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,0), 1)
    # Write annotated frame to disk asynchronously to avoid blocking gaze detection
    try:
        def _write_image(img, path):
            try:
                cv2.imwrite(path, img)
            except Exception:
                logging.exception("Failed to write annotated frame to %s", path)

        # submit to executor; returns a Future we intentionally ignore
        _executor.submit(_write_image, annotated_frame.copy(), image_file)
    except Exception:
        # fallback to synchronous write on failure
        logging.exception("Async write failed, falling back to synchronous write")
        cv2.imwrite(image_file, annotated_frame)

    print(f"GAZE ATTENTION: {attention_percent}")
    # Determine whether multiple faces are present for this frame (if function available)
    one_face = False
    try:
        # are_there_multiple_faces is defined in this module
        one_face = is_there_one_face(image)
    except Exception:
        # If face cascade or detection is not available, default to False and log
        logging.exception("Failed to determine face count for frame %s", frame_counter)

    # Return both attention percentage and multiple-face flag
    return attention_percent, one_face