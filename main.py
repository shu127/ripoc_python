# -*- encoding: utf-8 -*-
import cv2
import os
import matplotlib
import matplotlib.pyplot as plt
import time

import ripoc

if __name__ == "__main__":    
    # VS solution folder home
    dirPath = os.path.dirname(__file__)
    dirPath = os.path.join(dirPath.rstrip(os.path.basename(dirPath)), "image")

    baseImageFile = ""
    targetImageFile = ""
    
    imgBase = cv2.imread(os.path.join(dirPath, baseImageFile), cv2.IMREAD_UNCHANGED)
    imgTarget = cv2.imread(os.path.join(dirPath, targetImageFile), cv2.IMREAD_UNCHANGED)
    
    figSrc, (figSrcLeft, figSrcRight) = plt.subplots(ncols=2)
    figSrcLeft.imshow(imgBase)
    figSrcRight.imshow(imgTarget)
#    plt.show(figSrc)

    angle, scale, ret, resp = ripoc.ripoc(imgBase, imgTarget, 85, True)
#    start = time.clock()
#    angle, scale, ret, resp = ripoc.ripoc(imgBase, imgTarget, 85, False)
#    stop = time.clock()
#    print("Elapsed Time : " + str(stop - start) + " sec.")

    center = (imgBase.shape[0] / 2.0, imgBase.shape[1] / 2.0)
    matTrans = cv2.getRotationMatrix2D(center, angle, scale)
    imgTrans = cv2.warpAffine(imgBase, matTrans, imgBase.shape[0:2], flags=cv2.INTER_LINEAR)
    cv2.imwrite(os.path.join(dirPath, "estimate.png"), imgTrans)
    
    print("angle = " + str(angle) + "\nscale = " + str(scale))
    print("ret = " + str(ret) + "\nresponse = " + str(resp))
