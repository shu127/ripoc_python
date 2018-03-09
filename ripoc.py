# -*- encoding: utf-8 -*-
# References
# https://www.daisukekobayashi.com/blog/rotation-invariant-phase-only-correlation-in-python/
# https://stackoverflow.com/questions/14132951/how-to-obtain-the-scale-and-rotation-angle-from-logpolar-transform

import cv2
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def _logPolar(srcImg, center, mag):
    def _Bicubic(px, py):
        if (1.0 < px) and (px < (srcImg.shape[1] - 2.0)) and (1.0 < py) and (py < (srcImg.shape[0] - 2.0)):
#        px = 1.0 + sys.float_info.epsilon if sx < 1.0 else ((srcImg.shape[1] - 3.0) if sx > (srcImg.shape[1] - 3.0 - sys.float_info.epsilon) else sx)
#        py = 1.0 + sys.float_info.epsilon if sy < 1.0 else ((srcImg.shape[0] - 3.0) if sy > (srcImg.shape[0] - 3.0 - sys.float_info.epsilon) else sy)
#            roi = np.matrix([[srcImg[int(py) + i, int(px) + j] for j in range(-1, 3)] for i in range(-1, 3)])
#            hx = np.matrix([[_h( abs(abs(px - int(px)) - i) )] for i in range(-1, 3)])
#            hy = np.matrix([ _h( abs(abs(py - int(py)) - i) )  for i in range(-1, 3)])
            roi = srcImg[int(py)-1:int(py)+3, int(px)-1:int(px)+3]
            hx = np.matrix([[_h(px - int(px) + 1)], [_h(px - int(px))], [_h(int(px) - px + 1)], [_h(int(px) - px + 2)]])
            hy = np.matrix([_h(py - int(py) + 1), _h(py - int(py)), _h(int(py) - py + 1), _h(int(py) - py + 2)])
            return (hy * roi * hx)[0,0]
        return 0.0

    def _h(t):
        a = -0.2
        if t > 2.0:
            return 0.0
        if t <= 1.0:
            return (a + 2.0) * (t ** 3) - (a + 3.0) * (t ** 2) + 1.0 
        return a * (t ** 3) - 5.0 * a * (t ** 2) + 8.0 * a * t - 4.0 * a

    rho = [np.exp( float(i) / mag ) for i in range(0, srcImg.shape[1])]
#    theta = [2.0 * np.pi * i / float(srcImg.shape[0]) for i in reversed(range(0, srcImg.shape[0]))]
    theta = [2.0 * np.pi * i / float(srcImg.shape[0]) for i in range(0, srcImg.shape[0])]
    lpImg = np.matrix([[_Bicubic(j * np.sin(i) + center[1], j * np.cos(i) + center[0]) for j in rho] for i in theta])
    return lpImg

def ripoc(srcImg1, srcImg2, logpolarMagnitude=40, plotFig=False):
    grayImg1 = np.float64( cv2.cvtColor(srcImg1, cv2.COLOR_BGRA2GRAY) ) / 255.0 
    grayImg2 = np.float64( cv2.cvtColor(srcImg2, cv2.COLOR_BGRA2GRAY) ) / 255.0

    center = (grayImg1.shape[0] / 2.0, grayImg1.shape[1] / 2.0)
#    lpImg1 = cv2.logPolar(grayImg1, center, logpolarMagnitude, cv2.INTER_CUBIC + cv2.WARP_FILL_OUTLIERS)
    lpImg1 = _logPolar(grayImg1, center, logpolarMagnitude)
    center = (grayImg2.shape[0] / 2.0, grayImg2.shape[1] / 2.0)
#    lpImg2 = cv2.logPolar(grayImg2, center, logpolarMagnitude, cv2.INTER_CUBIC + cv2.WARP_FILL_OUTLIERS)
    lpImg2 = _logPolar(grayImg2, center, logpolarMagnitude)

    if plotFig:
        figLP, (figLPLeft, figLPRight) = plt.subplots(ncols=2)
        figLPLeft.imshow(lpImg1, cmap="gray")
        figLPRight.imshow(lpImg2, cmap="gray")
        plt.show(figLP)

    hann = cv2.createHanningWindow(lpImg1.shape, cv2.CV_64F)
    ret, resp = cv2.phaseCorrelate(lpImg1, lpImg2, hann)

#    angle = -((ret[1] + 0.5) * 360.0 / float(lpImg1.shape[0]) )
#    scale = np.exp(ret[0] / logpolarMagnitude)
    angle = ret[1] * 360.0 / float(lpImg1.shape[0])
    scale = np.exp(ret[0] / logpolarMagnitude)
#    print("ret = " + str(ret) + "\nresponse = " + str(resp))
#    print("angle = " + str(angle) + "\nscale = " + str(scale))

    matTrans = cv2.getRotationMatrix2D(center, angle, scale)
    imgTrans = cv2.warpAffine(srcImg1, matTrans, srcImg1.shape[0:2], flags=cv2.INTER_LINEAR)
    if plotFig:
        figDst, (figDstLeft, figDstRight) = plt.subplots(ncols=2)
        figDstLeft.imshow(imgTrans)
        figDstRight.imshow(srcImg2)
        plt.show(figDst)

        subtractImg = cv2.subtract(cv2.cvtColor(imgTrans,cv2.COLOR_BGRA2RGB), cv2.cvtColor(srcImg2,cv2.COLOR_BGRA2RGB))
        plt.imshow(subtractImg)
        plt.show()

    grayTrans = np.float64( cv2.cvtColor(imgTrans, cv2.COLOR_BGRA2GRAY) ) / 255.0 
    ret, resp = cv2.phaseCorrelate(grayTrans, grayImg2, hann)
    return angle, scale, ret, resp
