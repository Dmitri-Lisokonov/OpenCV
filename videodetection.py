import cv2
import numpy as np
import imutils


def capture_camera():
    cap = cv2.VideoCapture(0)
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Display the resulting frame
        cv2.imshow('frame', gray)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


def detect_face():
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')

    while True:
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.imshow('img', img)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    cv2.destroyAllWindows()
    cap.release()


def detect_movement():
    cap = cv2.VideoCapture(0)

    ret, frame1 = cap.read()
    ret, frame2 = cap.read()

    while cap.isOpened():
        diff = cv2.absdiff(frame1, frame2)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=3)
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)

            if cv2.contourArea(contour) < 10000:
                continue
            cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame1, 'Movement', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('feed', frame1)
        frame1 = frame2
        ret, frame2 = cap.read()
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    cv2.destroyAllWindows()
    cap.release()


def detect_object_coordinates():
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    while True:
        _, frame = cap.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([90, 60, 0])
        upper_blue = np.array([121, 255, 255])

        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        cnts = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        for c in cnts:
            area = cv2.contourArea(c)
            cv2.drawContours(frame, [c], -1, (0, 255, 0), 3)
            m = cv2.moments(c)
            try:
                if m["m10"] != 0 and m["m00"] != 0 and m["m01"] != 0 and m["m00"] != 0:
                    cx = int(m["m10"] / m["m00"])
                    cy = int(m["m01"] / m["m00"])
                    cv2.circle(frame, (cx, cy), 7, (255, 255, 255), -1)
                    cv2.putText(frame, f'X = {cx}, Y = {cy}', (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            except Exception as e:
                print(e)
                continue

        cv2.imshow("Object coordinate detection", frame)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    cv2.destroyAllWindows()
    cap.release()
