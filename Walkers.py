from cgitb import grey
import cv2
bodyClassifier=cv2.CascadeClassifier('PEDESTRIAN-DETECTOR-main/haarcascade_fullbody.xml')
vid=cv2.VideoCapture('PEDESTRIAN-DETECTOR-main/walking.avi')
while True:
    ret,frame=vid.read()
    grey=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    bodies=bodyClassifier.detectMultiScale(grey,1.2,3)
    for (x,y,w,h) in bodies:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.imshow('pedestestrains',frame)
    if cv2.waitKey(1)==32:
        break
vid.release()
cv2.destroyAllWindows()