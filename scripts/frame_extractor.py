import cv2
import os

cam = cv2.VideoCapture("video.mp4")

try:
    if not os.path.exists('data'):
        os.makedirs('data')
except OSError:
    print('Error: Creating directory of data')

currentframe = 0

while (True):
    ret, frame = cam.read()

    if ret:
        currentframe += 1 
        if(currentframe % 4 != 0): # Get one frame in every four frames.
            continue
        name = './data/frame' + str(currentframe) + '.jpg'
        cv2.imwrite(name, frame)
    else:
        break
cam.release()
cv2.destroyAllWindows()
