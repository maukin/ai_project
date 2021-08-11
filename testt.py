import cv2
import numpy as np
import pafy
import youtube_dl

url = "https://www.youtube.com/watch?v=FnZfwhNMCTQ"
videoPafy = pafy.new(url)
best = videoPafy.getbest(preftype="any")

video = cv2.VideoCapture(best.url)


#
object_detection = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=200) #поменять varThreshold



while True:
    ret, frame = video.read() #успешное открытие файла

    # frame = cv2.GaussianBlur(frame, (13, 13), 0)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)




    height = frame.shape[1]
    width = frame.shape[0]
    #height, width, _ = frame.shape
    # print(height, width)

    #Extract region of interest
    roi = frame [400:720, 300:964]
    #roi = frame[:, :]

    #Object Detection
    mask = object_detection.apply(roi) #(frame)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        # Calc area and remove small elements
        area = cv2.contourArea(cnt)
        if area > 3500:
            # cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 1) #(frame)
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)


    cv2.imshow('Traffic', frame)
    cv2.imshow("Mask", mask)
    cv2.imshow("ROI", roi)


    if cv2.waitKey(30) & 0xFF == ord('q'): #если видео вопсроизводится и нажата клавиша q
        break

video.release()
cv2.destroyAllWindows()