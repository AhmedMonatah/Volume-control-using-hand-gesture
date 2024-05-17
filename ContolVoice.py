import cv2
import mediapipe as mp
from math import hypot
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import numpy as np

# Initialize camera
cap = cv2.VideoCapture(0)

# Initialize hand detection
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

# Initialize audio control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volbar = 100
volper = 0
volMin, volMax = volume.GetVolumeRange()[:2]

# Initialize face detection
mpFace = mp.solutions.face_detection
face_detection = mpFace.FaceDetection()

while True:
    # Capture frame from camera
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Detect hand landmarks
    results = hands.process(imgRGB)

    lmList = []
    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            for id, lm in enumerate(hand_landmark.landmark):
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
            mpDraw.draw_landmarks(img, hand_landmark, mpHands.HAND_CONNECTIONS)

    # Detect face and determine if eyes are open or closed
    results_face = face_detection.process(imgRGB)
    if results_face.detections:
        for detection in results_face.detections:
            bbox = detection.location_data.relative_bounding_box
            x, y, w, h = int(bbox.xmin * img.shape[1]), int(bbox.ymin * img.shape[0]), \
                         int(bbox.width * img.shape[1]), int(bbox.height * img.shape[0])
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

            if lmList:
                    x1, y1 = lmList[4][1], lmList[4][2]  # Thumb
                    x2, y2 = lmList[8][1], lmList[8][2]  # Index finger

                    cv2.circle(img, (x1, y1), 13, (255, 0, 0), cv2.FILLED)
                    cv2.circle(img, (x2, y2), 13, (255, 0, 0), cv2.FILLED)
                    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)

                    length = hypot(x2 - x1, y2 - y1)
                    vol = np.interp(length, [30, 350], [volMin, volMax])
                    volbar = np.interp(length, [30, 350], [400, 150])
                    volper = np.interp(length, [30, 350], [0, 100])

                    volume.SetMasterVolumeLevel(vol, None)

                    cv2.rectangle(img, (50, 150), (85, 400), (0, 0, 255), 4)
                    cv2.rectangle(img, (50, int(volbar)), (85, 400), (0, 0, 255), cv2.FILLED)
                    cv2.putText(img, f"{int(volper)}%", (10, 40), cv2.FONT_ITALIC, 1, (0, 255, 98),3)

    else:
        # No face detected, do not perform volume control
        cv2.putText(img, "No Face Detected", (10, 40), cv2.FONT_ITALIC, 1, (0, 0, 255), 3)

    cv2.imshow('Image', img)

    if cv2.waitKey(1) & 0xFF == ord(' '):
        break

cap.release()
cv2.destroyAllWindows()
