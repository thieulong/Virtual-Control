import cv2
import mediapipe
import math
from subprocess import call
 
drawingModule = mediapipe.solutions.drawing_utils
handsModule = mediapipe.solutions.hands
 
capture = cv2.VideoCapture(3)
 
frameWidth = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
frameHeight = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

def distance(a, b):
    return math.sqrt(math.pow((b[0]-a[0]),2)+math.pow(b[1]-a[1],2))

def midpoint(a, b):
    return ((a[0]+b[0])/2, (a[1]+b[1])/2)
 
with handsModule.Hands(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=1) as hands:
 
    while True:
 
        ret, frame = capture.read()
 
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
 
        if results.multi_hand_landmarks != None:
            for handLandmarks in results.multi_hand_landmarks:
                # Finger tip 
                fingertip = 8
                fingertip_normalizedLandmark = handLandmarks.landmark[fingertip]
                fingertip_pixelCoordinatesLandmark = drawingModule._normalized_to_pixel_coordinates(fingertip_normalizedLandmark.x, fingertip_normalizedLandmark.y, frameWidth, frameHeight)
                cv2.circle(frame, fingertip_pixelCoordinatesLandmark, 5, (0, 255, 0), -1)

                # Thumb 
                thumb = 4
                thumb_normalizedLandmark = handLandmarks.landmark[thumb]
                thumb_pixelCoordinatesLandmark = drawingModule._normalized_to_pixel_coordinates(thumb_normalizedLandmark.x, thumb_normalizedLandmark.y, frameWidth, frameHeight)
                cv2.circle(frame, thumb_pixelCoordinatesLandmark, 5, (0, 255, 0), -1)

                cv2.line(frame, fingertip_pixelCoordinatesLandmark, thumb_pixelCoordinatesLandmark, (0, 255, 0), 2)

                if fingertip_pixelCoordinatesLandmark and thumb_pixelCoordinatesLandmark:
                    distance_finger_thumb = distance(a=fingertip_pixelCoordinatesLandmark, b=thumb_pixelCoordinatesLandmark)
                    volume = int(distance_finger_thumb-20)
                    if volume >= 100: volume = 100
                    elif volume<=0: volume = 0
                    call(["amixer", "-D", "pulse", "sset", "Master", "{volume}%".format(volume=volume)])

                    cv2.putText(frame, "Volume: {volume}%".format(volume=volume), (40,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 153), 2)
 
        cv2.imshow('Test hand', frame)
 
        if cv2.waitKey(1) == 32:
            break
 
cv2.destroyAllWindows()
capture.release()