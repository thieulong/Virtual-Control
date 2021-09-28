import cv2
import pyautogui
import time

hand_cascade = cv2.CascadeClassifier('mouse.xml')
fist_cascade = cv2.CascadeClassifier('click.xml')

camera = cv2.VideoCapture(1)

while True:

    ret, frame = camera.read()

    if ret:

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        down = hand_cascade.detectMultiScale(gray, 1.1, 12, minSize=(50, 50))
        up = fist_cascade.detectMultiScale(gray, 1.1, 12, minSize=(10, 10))

        for (x, y, w, h) in down:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
            pyautogui.press('down')

        for (x, y, w, h) in up:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3)
            pyautogui.press('up')

        # cv2.namedWindow("Virtual Scroll", cv2.WND_PROP_FULLSCREEN)
        # cv2.setWindowProperty("Virtual Scroll", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        # cv2.imshow("Virtual Scroll", frame)

    if cv2.waitKey(1) & 0xFF == 32:
        break

camera.release()
cv2.destroyAllWindows()
