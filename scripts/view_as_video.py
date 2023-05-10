import cv2
import glob
import time

for i in glob.glob("*.jpg"):
    img = cv2.imread(i)
    cv2.imshow("new", img)
    cv2.waitKey(1)
    time.sleep(0.025)
cv2.destroyAllWindows()