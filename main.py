import numpy as np
import cv2 as cv

def resize(frame):
    escala = .5
    return cv.resize(frame, 
                     None, 
                     fx=escala, 
                     fy=escala, 
                     interpolation=cv.INTER_AREA)

cap = cv.VideoCapture("carro.MP4")
ret, frame1 = cap.read()

frame1 = resize(frame1)

previo = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255

ret = True
while(ret):
    ret, frame2 = cap.read()
    
    if frame2 is None:
        break

    frame2 = resize(frame2)

    proximo = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)

    flow = cv.calcOpticalFlowFarneback(previo,
                                       proximo,
                                       None,
                                       0.5,
                                       3,
                                       15, 
                                       3, 
                                       5,
                                       1.2,
                                       0)
    
    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

    cv.imshow('frame2', bgr)

    k = cv.waitKey(30) & 0xff
    if k == 27:
        break

    previo = proximo

cv.destroyAllWindows()