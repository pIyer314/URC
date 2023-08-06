import cv2 as cv 
import numpy as np

#Function to rescale image to screen resolution
def redim(frame, scalew =2,scaleh =2): 
    w = int(frame.shape[1]*scalew)
    h = int(frame.shape[0]*scaleh)
    dim = (w,h) 
    return cv.resize(frame, dim, interpolation=cv.INTER_AREA)

#Dictionary of all predefined Aruco Dictionaries
ARUCO_DICT = {
	"DICT_4X4_50": cv.aruco.DICT_4X4_50,
	"DICT_4X4_100": cv.aruco.DICT_4X4_100,
	"DICT_4X4_250": cv.aruco.DICT_4X4_250,
	"DICT_4X4_1000": cv.aruco.DICT_4X4_1000,
	"DICT_5X5_50": cv.aruco.DICT_5X5_50,
	"DICT_5X5_100": cv.aruco.DICT_5X5_100,
	"DICT_5X5_250": cv.aruco.DICT_5X5_250,
	"DICT_5X5_1000": cv.aruco.DICT_5X5_1000,
	"DICT_6X6_50": cv.aruco.DICT_6X6_50,
	"DICT_6X6_100": cv.aruco.DICT_6X6_100,
	"DICT_6X6_250": cv.aruco.DICT_6X6_250,
	"DICT_6X6_1000": cv.aruco.DICT_6X6_1000,
	"DICT_7X7_50": cv.aruco.DICT_7X7_50,
	"DICT_7X7_100": cv.aruco.DICT_7X7_100,
	"DICT_7X7_250": cv.aruco.DICT_7X7_250,
	"DICT_7X7_1000": cv.aruco.DICT_7X7_1000,
	"DICT_ARUCO_ORIGINAL": cv.aruco.DICT_ARUCO_ORIGINAL,
	"DICT_APRILTAG_16h5": cv.aruco.DICT_APRILTAG_16h5,
	"DICT_APRILTAG_25h9": cv.aruco.DICT_APRILTAG_25h9,
	"DICT_APRILTAG_36h10": cv.aruco.DICT_APRILTAG_36h10,
	"DICT_APRILTAG_36h11": cv.aruco.DICT_APRILTAG_36h11
}

#Loading and rescaling image from path
path  = input("Path :")
path.replace("\\", "/")
img = cv.imread(path)
img = redim(img,scalew=0.75,scaleh=0.75)

#Identifying the different arUcotypes(predefined set only) present in the frame and stores it to arType
arType = []
for (arucoName, arucoDict) in ARUCO_DICT.items():

    arD = cv.aruco.getPredefinedDictionary(arucoDict)
    arP = cv.aruco.DetectorParameters()
    arDt = cv.aruco.ArucoDetector(arD, arP)
    (corners, ids, rejected) =arDt.detectMarkers(img)

    if len(corners) > 0:
        arType.append(arucoName)

arType =[*set(arType)]

#Detecting all aruco markers by iterating over the types of arUco identified in arType
for ar in arType :
    arD = cv.aruco.getPredefinedDictionary(ARUCO_DICT[ar])
    arP = cv.aruco.DetectorParameters()
    arDt = cv.aruco.ArucoDetector(arD, arP)
    (corners, ids, rejected) = arDt.detectMarkers(img)

    print("Type {} markers  ".format(ar[5:]))
    #Marking and labelling the arUco markers
    if len(corners) > 0:
        ids = ids.flatten()

        for (markerCorner, markerID) in zip(corners, ids):
            corners = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners            
            c2 = (int(bottomRight[0]), int(bottomRight[1]))            
            c1 = (int(topLeft[0]), int(topLeft[1]))
            c3 =(int((topRight[0]+topLeft[0])/2),int((bottomRight[1]+topRight[1])/2))
            cv.rectangle(img, c1,c2,(0,255,0),2)
            cv.putText(img, str(markerID),c3, cv.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 255), 1)
            print("id : {} ".format(str(markerID)))
    
cv.imshow("Detected Marker", img)
cv.waitKey(0)