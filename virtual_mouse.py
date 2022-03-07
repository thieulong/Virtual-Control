import cv2
import mediapipe
import pyautogui
import math

width, height = pyautogui.size()

print("Screen resolution:", width, "x", height)
 
drawingModule = mediapipe.solutions.drawing_utils
handsModule = mediapipe.solutions.hands
 
capture = cv2.VideoCapture(1)
 
frameWidth = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
frameHeight = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
 
 
with handsModule.Hands(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=1) as hands:
 
    while True:
 
        ret, frame = capture.read()
 
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
 
        if results.multi_hand_landmarks != None:
            for handLandmarks in results.multi_hand_landmarks:

                    # Cursor
                    fingertip = 8
                    fingertip_normalizedLandmark = handLandmarks.landmark[fingertip]
                    fingertip_pixelCoordinatesLandmark = drawingModule._normalized_to_pixel_coordinates(fingertip_normalizedLandmark.x, fingertip_normalizedLandmark.y, frameWidth, frameHeight)
 
                    cv2.circle(frame, fingertip_pixelCoordinatesLandmark, 5, (0, 255, 0), -1)

                    if fingertip_pixelCoordinatesLandmark:
                        x_cord = fingertip_pixelCoordinatesLandmark[0]*3
                        y_cord = fingertip_pixelCoordinatesLandmark[1]*2.25
                        fingertip_cord = (1920 - x_cord, y_cord)

                        pyautogui.moveTo(fingertip_cord)

                    # # Click
                    thumbtip = 4
                    thumbtip_normalizedLandmark = handLandmarks.landmark[thumbtip]
                    thumbtip_pixelCoordinatesLandmark = drawingModule._normalized_to_pixel_coordinates(thumbtip_normalizedLandmark.x, thumbtip_normalizedLandmark.y, frameWidth, frameHeight)
 
                    cv2.circle(frame, thumbtip_pixelCoordinatesLandmark, 5, (0, 255, 0), -1)

                    distance_thumb_finger = math.sqrt(math.pow((thumbtip_pixelCoordinatesLandmark[0]-fingertip_pixelCoordinatesLandmark[0]),2)+math.pow((thumbtip_pixelCoordinatesLandmark[1]-fingertip_pixelCoordinatesLandmark[1]),2))

                    print(distance_thumb_finger)

                    if distance_thumb_finger <= 30:
                        pyautogui.mouseDown(button='left')
                    if distance_thumb_finger > 30:
                        
                        pyautogui.mouseUp(button='left')

 
        # cv2.imshow("Virtual Mouse", frame)
 
        if cv2.waitKey(1) == 27:
            break
 
cv2.destroyAllWindows()
capture.release()