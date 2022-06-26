import cv2
import numpy as np

vidcap = cv2.VideoCapture('datasets/IEMOCAP_full_release/Session1/dialog/avi/DivX/Ses01F_impro01.avi')
success,image = vidcap.read()
count = 0
while success:
    #y:y+h, x:x+w
    cropped_image = image[122:360, 6:354]
    #cropped_image2 = image[122:360, 6:354]
    cv2.imwrite("output/Ses01F_impro01_frame_female_%d.jpg" % count, cropped_image)
    success,image = vidcap.read()
    count += 1

print('Done: ', count)
