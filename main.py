from Data.DataLoader import DataLoader
from Data.Util import Util
import numpy as np
import cv2
import math

inputImageName = "lbj.png"
inWidth = 500
inHeight = 500

dataLoader = DataLoader(inWidth, inHeight, inputImageName)
inputContourPoints = dataLoader.getContourPoints()
posePoints = dataLoader.posePoints
poseLines = dataLoader.poseLines

workImage = np.zeros((inHeight, inWidth, 3), dtype = np.uint8)
util = Util(inWidth, inHeight, workImage, dataLoader.inputImageResize)
inputContourPointsRefine = []
for i in range(0, len(inputContourPoints) - 1):
  for point in util.getInterpolatePoints(inputContourPoints[i], inputContourPoints[i+1]):
    inputContourPointsRefine.append(point)
    


bodyContour = util.getBodyContour([poseLines[3], poseLines[5], poseLines[8], poseLines[11]], inputContourPointsRefine)
util.drawContour(bodyContour, workImage, dataLoader.inputImageResize)
# for point in inputContourPointsRefine:
#     ww, hh = point
#     workImage[hh, ww] = (0, 0, 255)
#util.drawContour(inputContourPointsRefine, workImage, dataLoader.inputImageResize)    
 
contourPointsMap1Refine, contourPointsMap2Refine, contourPointsMap3Refine = util.deform(inputContourPointsRefine, poseLines[3], poseLines[4], math.pi/6, math.pi/2.3, True)  
for point in contourPointsMap1Refine:
    ww, hh = point
    workImage[hh, ww] = (0, 0, 255)
for point in contourPointsMap2Refine:
    ww, hh = point
    workImage[hh, ww] = (0, 255, 0)
for point in contourPointsMap3Refine:
    ww, hh = point
    workImage[hh, ww] = (255, 0, 0)
    
contourPointsMap1Refine, contourPointsMap2Refine, contourPointsMap3Refine = util.deform(inputContourPointsRefine, poseLines[5], poseLines[6], -math.pi/8, -math.pi/2, False)  
for point in contourPointsMap1Refine:
    ww, hh = point
    workImage[hh, ww] = (0, 0, 255)
for point in contourPointsMap2Refine:
    ww, hh = point
    workImage[hh, ww] = (0, 255, 0)
for point in contourPointsMap3Refine:
    ww, hh = point
    workImage[hh, ww] = (255, 0, 0)
    
contourPointsMap1Refine, contourPointsMap2Refine, contourPointsMap3Refine = util.deform(inputContourPointsRefine, poseLines[8], poseLines[9], -math.pi/12, -math.pi/12, True)  
for point in contourPointsMap1Refine:
    ww, hh = point
    workImage[hh, ww] = (0, 0, 255)
for point in contourPointsMap2Refine:
    ww, hh = point
    workImage[hh, ww] = (0, 255, 0)
for point in contourPointsMap3Refine:
    ww, hh = point
    workImage[hh, ww] = (255, 0, 0)
    
contourPointsMap1Refine, contourPointsMap2Refine, contourPointsMap3Refine = util.deform(inputContourPointsRefine, poseLines[11], poseLines[12], math.pi/3, -math.pi/4, False)  
for point in contourPointsMap1Refine:
    ww, hh = point
    workImage[hh, ww] = (0, 0, 255)
for point in contourPointsMap2Refine:
    ww, hh = point
    workImage[hh, ww] = (0, 255, 0)
for point in contourPointsMap3Refine:
    ww, hh = point
    workImage[hh, ww] = (255, 0, 0)
    
cv2.imshow('', workImage)
cv2.waitKey(0)





