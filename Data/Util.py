import numpy as np
import math
from numpy.linalg import inv

class Util:
    def __init__(self, inWidth, inHeight, workImage, inputImage):
        self.inWidth = inWidth
        self.inHeight = inHeight
        self.cutContours = [[]] * 8
        self.workImage = workImage
        self.inputImage = inputImage
        
    def findStartAndEnd(self, contourPoints, targetPoint, startPoint, endPoint):
      w, h = targetPoint
      wStart, hStart = startPoint
      wEnd, hEnd = endPoint
    
      left = -1
      right = self.inWidth
      startIndex = -1
      endIndex = -1
    
      index90Degree = []
    
      for i in range(len(contourPoints)):
        point = contourPoints[i]
        wPoint, hPoint = point
    
        length1 = math.sqrt((hEnd - hStart) ** 2 + (wEnd - wStart) ** 2)
        length2 = math.sqrt((hPoint - h) ** 2 + (wPoint - w) ** 2)
        if length2 == 0:
          continue
        dotProduct = (hEnd - hStart) * (hPoint - h) + (wEnd - wStart) * (wPoint - w)
        angle = abs(np.arccos(dotProduct / length1 / length2))
        if abs(angle - math.pi / 2) < math.pi / 15:
          if startIndex == -1:
            startIndex = i
          else:
            endIndex = i
        else:
          if startIndex != -1 and endIndex != -1:
            index90Degree.append((startIndex, endIndex))
            startIndex = -1
            endIndex = -1
  
      # the 90 degree cut for left leg can include the right leg
      index90DegreeRefine = []
      left = -1
      right = self.inWidth
      index90DegreeLeft = -1, -1
      index90DegreeRight = -1, -1
      for pair in index90Degree:
        start2, end2 = pair
        ww, hh = contourPoints[start2]
        if ww < w and ww > left:
          left = contourPoints[start2][0]
          index90DegreeLeft = pair
        if ww > w and ww < right:
          right = contourPoints[start2][0]
          index90DegreeRight = pair
    
      index90DegreeRefine = [index90DegreeLeft, index90DegreeRight] if index90DegreeLeft[0] < index90DegreeRight[0] else [index90DegreeRight, index90DegreeLeft]
      return index90DegreeRefine
    
    def deform(self, contourPoints, lineUpper, lineLower, angleUpper, angleLower, isLeft):
        #upper control points
        lineUpperStart, lineUpperEnd = lineUpper      
        #find mid point of lineUpper
        wLineUpperMid = int((lineUpperStart[0] + lineUpperEnd[0]) / 2)
        hLineUpperMid = int((lineUpperStart[1] + lineUpperEnd[1]) / 2)
        
        indexA1, indexA2 = self.findStartAndEnd(contourPoints, (wLineUpperMid, hLineUpperMid), lineUpperStart, lineUpperEnd)
        pointA1 = contourPoints[indexA1[0]]
        pointA2 = contourPoints[indexA2[0]]
        
        #indexA3, indexA4 = self.findStartAndEnd(contourPoints, lineUpperEnd, lineUpperStart, lineUpperEnd)
        #pointA3 = contourPoints[indexA3[0]]
        #pointA4 = contourPoints[indexA4[0]]
        #A3 - A1 = end - start
        w3 = int((lineUpperEnd[0] - lineUpperStart[0]) / 2) + pointA1[0]
        h3 = int((lineUpperEnd[1] - lineUpperStart[1]) / 2) + pointA1[1]
        pointA3 = w3, h3
        w4 = int((lineUpperEnd[0] - lineUpperStart[0]) / 2) + pointA2[0]
        h4 = int((lineUpperEnd[1] - lineUpperStart[1]) / 2) + pointA2[1]
        pointA4 = w4, h4       
        
        #lower control points
        lineLowerStart, lineLowerEnd = lineLower
        #find mid point of lineUpper
        wLineLowerMid = int((lineLowerStart[0] + lineLowerEnd[0]) / 2)
        hLineLowerMid = int((lineLowerStart[1] + lineLowerEnd[1]) / 2)
        
        indexB1, indexB2 = self.findStartAndEnd(contourPoints, (wLineLowerMid, hLineLowerMid), lineLowerStart, lineLowerEnd)
        pointB1 = contourPoints[indexB1[0]]
        pointB2 = contourPoints[indexB2[0]]
        
        #indexB3, indexB4 = self.findStartAndEnd(contourPoints, lineLowerStart, lineLowerStart, lineLowerEnd)
        #pointB3 = contourPoints[indexB3[0]]
        #pointB4 = contourPoints[indexB4[0]]
        #B3 - B1 = start - end
        w3 = int((lineLowerStart[0] - lineLowerEnd[0]) / 2) + pointB1[0]
        h3 = int((lineLowerStart[1] - lineLowerEnd[1]) / 2) + pointB1[1]
        pointB3 = w3, h3
        w4 = int((lineLowerStart[0] - lineLowerEnd[0]) / 2) + pointB2[0]
        h4 = int((lineLowerStart[1] - lineLowerEnd[1]) / 2) + pointB2[1]
        pointB4 = w4, h4 
        
        #rotate
        lineUpperEndRotate = self.rotatePoint(lineUpperEnd, lineUpperStart, angleUpper)
        lineLowerEndRotate = self.rotatePoint(lineLowerEnd, lineUpperStart, angleUpper)
        lineLowerEndRotate = self.rotatePoint(lineLowerEndRotate, lineUpperEndRotate, angleLower)
        pointA1Rotate = self.rotatePoint(pointA1, lineUpperStart, angleUpper)
        pointA2Rotate = self.rotatePoint(pointA2, lineUpperStart, angleUpper)
        pointA3Rotate = self.rotatePoint(pointA3, lineUpperStart, angleUpper)
        pointA4Rotate = self.rotatePoint(pointA4, lineUpperStart, angleUpper)
        
        pointB1Rotate = self.rotatePoint(pointB1, lineUpperStart, angleUpper)
        pointB1Rotate = self.rotatePoint(pointB1Rotate, lineUpperEndRotate, angleLower)
        pointB2Rotate = self.rotatePoint(pointB2, lineUpperStart, angleUpper)
        pointB2Rotate = self.rotatePoint(pointB2Rotate, lineUpperEndRotate, angleLower)
        pointB3Rotate = self.rotatePoint(pointB3, lineUpperStart, angleUpper)
        pointB3Rotate = self.rotatePoint(pointB3Rotate, lineUpperEndRotate, angleLower)
        pointB4Rotate = self.rotatePoint(pointB4, lineUpperStart, angleUpper)        
        pointB4Rotate = self.rotatePoint(pointB4Rotate, lineUpperEndRotate, angleLower)
        
        indexUpperStart, indexUpperEnd = self.findStartAndEnd(contourPoints, lineUpperStart, lineUpperStart, lineUpperEnd) 
        
        cutPoints = self.getInterpolatePoints(contourPoints[indexUpperEnd[0]], contourPoints[indexUpperStart[0]])
        
        contourPointsMap1 = []
        contourPointsMap1.extend(cutPoints)
        
        controlPointsInput = [contourPoints[indexUpperEnd[0]], contourPoints[indexUpperStart[0]], pointA1]
        controlPointsOutput = [contourPoints[indexUpperEnd[0]], contourPoints[indexUpperStart[0]], pointA1Rotate]
        for index in range(indexUpperStart[0], indexA1[0]):
            point = contourPoints[index]
            #pointMap = self.rotatePoint(point, lineUpperStart, angleUpper)
            pointMap = self.mapPoint(point, controlPointsInput, controlPointsOutput)
            contourPointsMap1.append(pointMap)
        cutPoints2 = self.getInterpolatePoints(pointA1Rotate, pointA2Rotate)
        contourPointsMap1.extend(cutPoints2)
        
        controlPointsInput = [contourPoints[indexUpperEnd[0]], contourPoints[indexUpperStart[0]], pointA2]
        controlPointsOutput = [contourPoints[indexUpperEnd[0]], contourPoints[indexUpperStart[0]], pointA2Rotate]
        for index in range(indexA2[0], indexUpperEnd[0]):
            point = contourPoints[index]
            #pointMap = self.rotatePoint(point, lineUpperStart, angleUpper)
            pointMap = self.mapPoint(point, controlPointsInput, controlPointsOutput)
            contourPointsMap1.append(pointMap)
        contourPointsMap1Refine = []
        for i in range(0, len(contourPointsMap1) - 1):
            for point in self.getInterpolatePoints(contourPointsMap1[i], contourPointsMap1[i+1]):
                contourPointsMap1Refine.append(point)
        for point in self.getInterpolatePoints(contourPointsMap1[-1], contourPointsMap1[0]):
            contourPointsMap1Refine.append(point)
            
        controlPointsInput = [contourPoints[indexUpperEnd[0]], contourPoints[indexUpperStart[0]], pointA1, pointA2]
        controlPointsOutput = [contourPoints[indexUpperEnd[0]], contourPoints[indexUpperStart[0]], pointA1Rotate, pointA2Rotate]
        self.drawIt(contourPointsMap1Refine, controlPointsInput, controlPointsOutput, self.workImage, self.inputImage)
        
        controlPointsInput2 = [pointA1, pointA2, pointB1, pointB2]
        controlPointsOutput2 = [pointA1Rotate, pointA2Rotate, pointB1Rotate, pointB2Rotate]
        contourPointsMap2 = []
        cutPoints3 = self.getInterpolatePoints(pointA2Rotate, pointA1Rotate)
        contourPointsMap2.extend(cutPoints3)
        pointA3B3Mid = (pointA3[0] + pointB3[0]) / 2, (pointA3[1] + pointB3[1]) / 2
        pointA3B3RotateMid = (pointA3Rotate[0] + pointB3Rotate[0]) / 2, (pointA3Rotate[1] + pointB3Rotate[1]) / 2
        #controlPointsInput = [pointA1, pointA3B3Mid, pointB1] if isLeft else [pointA1, pointA3, pointB3, pointB1]
        controlPointsInput = self.createBezierCurve([pointA1, pointA3B3Mid, pointB1]) if isLeft else self.createBezierCurve([pointA1, pointA3, pointB3, pointB1])
        
        #controlPointsInputRotate = [pointA1Rotate, pointA3B3RotateMid, pointB1Rotate] if isLeft else [pointA1Rotate, pointA3Rotate, pointB3Rotate, pointB1Rotate]
        #controlPointsInputRotate = [pointA1Rotate, pointA3B3RotateMid, pointB1Rotate] if isLeft else self.createCurve(pointA1Rotate, pointA3Rotate, pointB3Rotate, pointB1Rotate, False)
        controlPointsInputRotate = self.createBezierCurve([pointA1Rotate, pointA3B3RotateMid, pointB1Rotate]) if isLeft else self.createBezierCurve([pointA1Rotate, pointA3Rotate, pointB3Rotate, pointB1Rotate])
        
        for index in range(indexA1[0], indexB1[0]):
            point = contourPoints[index]
            pointMap = self.mapPoint(point, controlPointsInput, controlPointsInputRotate)
            contourPointsMap2.append(pointMap)
        cutPoints4 = self.getInterpolatePoints(pointB1Rotate, pointB2Rotate)
        contourPointsMap2.extend(cutPoints4)
        pointA4B4Mid = (pointA4[0] + pointB4[0]) / 2, (pointA4[1] + pointB4[1]) / 2
        pointA4B4RotateMid = (pointA4Rotate[0] + pointB4Rotate[0]) / 2, (pointA4Rotate[1] + pointB4Rotate[1]) / 2
        #controlPointsInput = [pointA2, pointA4B4Mid, pointB4] if not isLeft else [pointA2, pointA4, pointB4, pointB2]
        controlPointsInput = self.createBezierCurve([pointA2, pointA4B4Mid, pointB4]) if not isLeft else self.createBezierCurve([pointA2, pointA4, pointB4, pointB2])
        #controlPointsInputRotate = [pointA2Rotate, pointA4B4RotateMid, pointB4Rotate] if not isLeft else [pointA2Rotate, pointA4Rotate, pointB4Rotate, pointB2Rotate]
        #controlPointsInputRotate = [pointA2Rotate, pointA4B4RotateMid, pointB4Rotate] if not isLeft else self.createCurve(pointA2Rotate, pointA4Rotate, pointB4Rotate, pointB2Rotate, True)
        controlPointsInputRotate = self.createBezierCurve([pointA2Rotate, pointA4B4RotateMid, pointB4Rotate]) if not isLeft else self.createBezierCurve([pointA2Rotate, pointA4Rotate, pointB4Rotate, pointB2Rotate])
        
        for index in range(indexB2[0], indexA2[0]):
            point = contourPoints[index]
            pointMap = self.mapPoint(point, controlPointsInput, controlPointsInputRotate)
            contourPointsMap2.append(pointMap)  
        
        contourPointsMap2Refine = []
        for i in range(0, len(contourPointsMap2) - 1):
            for point in self.getInterpolatePoints(contourPointsMap2[i], contourPointsMap2[i+1]):
                contourPointsMap2Refine.append(point)
        for point in self.getInterpolatePoints(contourPointsMap2[-1], contourPointsMap2[0]):
            contourPointsMap2Refine.append(point)
        self.drawIt(contourPointsMap2Refine, controlPointsInput2, controlPointsOutput2, self.workImage, self.inputImage)  
        
        controlPointsInput = [pointB1, pointB2, lineUpperEnd]
        controlPointsOutput = [pointB1Rotate, pointB2Rotate, lineUpperEndRotate]
        contourPointsMap3 = []
        cutPoints5 = self.getInterpolatePoints(pointB2Rotate, pointB1Rotate)
        contourPointsMap3.extend(cutPoints5)
        for index in range(indexB1[0], indexB2[0]):
            point = contourPoints[index]
            pointMap = self.rotatePoint(point, lineUpperStart, angleUpper)
            pointMap = self.rotatePoint(pointMap, lineUpperEndRotate, angleLower)
            contourPointsMap3.append(pointMap)
        contourPointsMap3Refine = []
        for i in range(0, len(contourPointsMap3) - 1):
            for point in self.getInterpolatePoints(contourPointsMap3[i], contourPointsMap3[i+1]):
                contourPointsMap3Refine.append(point)
        for point in self.getInterpolatePoints(contourPointsMap3[-1], contourPointsMap3[0]):
            contourPointsMap3Refine.append(point)
        self.drawUpperContour(contourPointsMap3Refine, controlPointsOutput, controlPointsInput)        
        
        return contourPointsMap1Refine, contourPointsMap2Refine, contourPointsMap3Refine

    def createBezierCurve(self, controlPoints):
        tList = []
        space = 20
        for i in range(space + 1):
            tList.append(float(i / space))
        
        curvePoints = []
        if len(controlPoints) == 2:
            w1, h1 = controlPoints[0]
            w2, h2 = controlPoints[1]
            for t in tList:
                w = w1 * (1 - t) + w2 * t
                h = h1 * (1 - t) + h2 * t
                curvePoints.append((w, h))
            return curvePoints
        
        points1 = self.createBezierCurve(controlPoints[0:-1])
        points2 = self.createBezierCurve(controlPoints[1:])
        
        for i in range(space + 1):
            t = tList[i]
            w1, h1 = points1[i]
            w2, h2 = points2[i]
            w = w1 * (1 - t) + w2 * t
            h = h1 * (1 - t) + h2 * t
            curvePoints.append((w, h))
        
        return curvePoints
    
    def drawIt(self, contourPointsMap, controlPointsInput, controlPointsOutput, workImage, inputImage):
        hMin = self.inHeight
        hMax = -1
        for point in contourPointsMap:
          w, h = point
          if h < hMin:
            hMin = h
          if h > hMax:
            hMax = h
      
        for h in range(hMin, hMax):
          wMin = self.inWidth
          wMax = -1
          for point in contourPointsMap:
            w, h2 = point
            if h2 == h:
              if w < wMin:
                wMin = w
              if w > wMax:
                wMax = w
        
          for w in range(wMin, wMax):
            ww, hh = self.mapPoint((w, h), controlPointsOutput, controlPointsInput)
            workImage[h, w] = inputImage[hh, ww]          
        
    
    def mapPoint(self, point, controlPointsInput, controlPointsOutput):
        weights = []
        sumDist = 0
        for pointInput in controlPointsInput:
            dist = self.distanceBetweenPoints(point, pointInput) + 0.001
            weights.append(1 / dist)
            sumDist += 1 / dist
        
        wMap = 0
        hMap = 0
        for i in range(len(weights)):
            wOutput, hOutput = controlPointsOutput[i]
            wMap += wOutput * weights[i] / sumDist
            hMap += hOutput * weights[i] / sumDist
        
        return int(wMap), int(hMap)
    
    def distanceBetweenPoints(self, point1, point2):
        w1, h1 = point1
        w2, h2 = point2
        
        return math.sqrt((h1 - h2) ** 2 + (w1 - w2) ** 2)
            
    def rotatePoint(self, point, center, angle):
        wCenter, hCenter = center
        w, h = point
        wRotate = int((w - wCenter) * math.cos(angle) - (h - hCenter) * math.sin(angle) + wCenter)
        hRotate = int((w - wCenter) * math.sin(angle) + (h - hCenter) * math.cos(angle) + hCenter)
        
        return (wRotate, hRotate)
    
    def getInterpolatePoints(self, pointStart, pointEnd):
      wStart, hStart = pointStart
      wEnd, hEnd = pointEnd
    
      points = []
    
      if abs(wStart - wEnd) > abs(hStart - hEnd):
        step =  1 if wStart < wEnd else -1
    
        for w in range(wStart, wEnd, step):
          k = (w - wStart) / (wEnd - wStart)
          h = round(hStart + k * (hEnd - hStart))
          points.append((w, h))
      else:
        step =  1 if hStart < hEnd else -1
    
        for h in range(hStart, hEnd, step):
          k = (h - hStart) / (hEnd - hStart)
          w = round(wStart + k * (wEnd - wStart))
          points.append((w, h))
    
      return points

    def getBodyContour(self, poseLines, contourPoints):
        def inInterval(target, start, end, length):
          if end - start < length / 2:
            return target > start and target < end
          else:
            return target > end or target < start

        intervals = []
        cutContours = []
        for poseLine in poseLines:
            pointPartStart, pointPartEnd = poseLine
            rangeStartAndEnd = self.findStartAndEnd(contourPoints, pointPartStart, pointPartStart, pointPartEnd)
            intervals.append((rangeStartAndEnd[0][0], rangeStartAndEnd[1][0]))
            cutContours.append(self.getInterpolatePoints(contourPoints[rangeStartAndEnd[0][0]], contourPoints[rangeStartAndEnd[1][0]]))
        
        trimmedBodyContour = []
        hasAddedParts = [False, False, False, False]
        for i in range(len(contourPoints)):
            isBody = True
            for j in range(len(hasAddedParts)):
                if inInterval(i, intervals[j][0], intervals[j][1], len(contourPoints)):
                    isBody = False
                    if not hasAddedParts[j]:
                        hasAddedParts[j] = True
                        trimmedBodyContour.extend(cutContours[j])
            if isBody:
                trimmedBodyContour.append(contourPoints[i])
             
        return trimmedBodyContour
     
    def drawContour(self, contour, workImage, inputImage):
      hMin = self.inHeight
      hMax = -1
      for point in contour:
        w, h = point
        if h < hMin:
          hMin = h
        if h > hMax:
          hMax = h
    
      for h in range(hMin, hMax):
        wMin = self.inWidth
        wMax = -1
        for point in contour:
          w, h2 = point
          if h2 == h:
            if w < wMin:
              wMin = w
            if w > wMax:
              wMax = w
        for w in range(wMin, wMax):
          workImage[h, w] = inputImage[h, w]         

    def drawUpperContour(self, contour, controlPointsOutput, controlPointsInput):
      hMin = self.inHeight
      hMax = -1
      for point in contour:
        w, h = point
        if h < hMin:
          hMin = h
        if h > hMax:
          hMax = h
    
      for h in range(hMin, hMax):
        wMin = self.inWidth
        wMax = -1
        for point in contour:
          w, h2 = point
          if h2 == h:
            if w < wMin:
              wMin = w
            if w > wMax:
              wMax = w
        for w in range(wMin, wMax):
          wInput, hInput = self.mapPoint2((w, h), controlPointsOutput, controlPointsInput)
          self.workImage[h, w] = self.inputImage[hInput, wInput]          

    def mapPoint2(self, point, controlPointsInput, controlPointsOutput):
        w, h = point
        w1Input, h1Input = controlPointsInput[0]
        w2Input, h2Input = controlPointsInput[1]
        w3Input, h3Input = controlPointsInput[2]
        w1Output, h1Output = controlPointsOutput[0]
        w2Output, h2Output = controlPointsOutput[1]
        w3Output, h3Output = controlPointsOutput[2]
        
        M = [[w1Input, w2Input, w3Input],
             [h1Input, h2Input, h3Input],
             [1, 1, 1]]
        MInv = inv(M)
        weights = np.matmul(MInv, [[w], [h], [1]])
        
        wMap = int(w1Output * weights[0][0] + w2Output * weights[1][0] + w3Output * weights[2][0])
        hMap = int(h1Output * weights[0][0] + h2Output * weights[1][0] + h3Output * weights[2][0])
        
        return wMap, hMap      
        
        
    
  
    
  
    
  
    
  