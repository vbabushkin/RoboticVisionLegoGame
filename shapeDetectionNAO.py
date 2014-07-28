__author__ = 'vahan'
import time
import cv
import cv2
import math
import numpy as np
import Image
import sys
from naoqi import ALProxy
from naoqi import ALBroker

BLUE_MIN = np.array([110, 50, 50],np.uint8)  #150 for red 110 for blue
BLUE_MAX = np.array([130, 255, 255],np.uint8)#180 for red 130 for blue

RED_MIN = np.array([150, 50, 50],np.uint8)  #150 for red 110 for blue
RED_MAX = np.array([180, 255, 255],np.uint8)#180 for red 130 for blue

YELLOW_MIN = np.array([10, 50, 50],np.uint8)  #150 for red 110 for blue
YELLOW_MAX = np.array([35, 255, 255],np.uint8)#180 for red 130 for blue

GREEN_MIN = np.array([40, 50, 50],np.uint8)
GREEN_MAX = np.array([80, 255, 255],np.uint8)

path = "/home/vahan/PycharmProjects/visionForLegoGame/images"

def convert2pil(img):
    cv_im = cv.CreateImageHeader(img.size, cv.IPL_DEPTH_8U, 3)
    r,g,b=img.split()
    pi2=Image.merge("RGB",(b,g,r))
    cv.SetData(cv_im, pi2.tobytes())
    return cv_im

def HSVColorValues(color):
    hsv_color = cv2.cvtColor(color,cv2.COLOR_BGR2HSV)
    return hsv_color


def getColour(IP, PORT):
    """
First get an image from Nao, then show it on the screen with PIL.
    :param IP:
    :param PORT:
"""


    myBroker = ALBroker("myBroker",
        "0.0.0.0", # listen to anyone
        0, # find a free port and use it
        IP, # parent broker IP
        PORT) # parent broker port

    camProxy = ALProxy("ALVideoDevice", IP, PORT)
    resolution = 2 # VGA
    colorSpace = 11 # RGB


    videoClient = camProxy.subscribe("python_client", resolution, colorSpace, 5)

    t0 = time.time()

    # Get a camera image.
    # image[6] contains the image data passed as an array of ASCII chars.
    naoImage = camProxy.getImageRemote(videoClient)

    t1 = time.time()

    # Time the image transfer.
    #print "Runde: ", b

    camProxy.unsubscribe(videoClient)


    # Now we work with the image returned and save it as a PNG using ImageDraw
    # package.

    # Get the image size and pixel array.
    imageWidth = naoImage[0]
    imageHeight = naoImage[1]
    array = naoImage[6]

    #Create a PIL Image Instance from our pixel array.
    img0= Image.frombytes("RGB", (imageWidth, imageHeight), array)


    #frame=np.asarray(convert2pil(img0)[:,:])

    #object_rect2=detectColor(img0, RED_MIN,RED_MAX)
    frame=detectShape(img0, RED_MIN,RED_MAX)

    #frame=selectDetected(object_rect1,frame)

    #frame=selectDetected(object_rect2,frame)
    # currentImage = path+ "/camImage1cm.jpg"
    # cv2.imwrite(currentImage, frame)
    cv2.imshow('contour',frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#to select detected objects to the main frame
def selectDetected(object_rect,frame):
    points = []
    box = cv2.cv.BoxPoints(object_rect)
    box = np.int0(box)
    minX=min(box[0][0],box[1][0],box[2][0],box[3][0])
    minY=min(box[0][1],box[1][1],box[2][1],box[3][1])
    maxX=max(box[0][0],box[1][0],box[2][0],box[3][0])
    maxY=max(box[0][1],box[1][1],box[2][1],box[3][1])
    center = (math.fabs(minX+(maxX-minX)/2.0), math.fabs(minY+(maxY-minY)/2.0))
    circleCenter=(int(center[0]), int(center[1]))
    cv2.drawContours(frame,[box],0,(0,0,255),2)
    cv2.circle(frame,circleCenter,3,(0, 255, 0),-1,8,0)
    return frame

def detectShape(image,COLOR_MIN,COLOR_MAX):
    #Convert to PIL image
    original=convert2pil(image)
    #convert to numpy array
    frame=np.asarray(original[:,:])
    #convert frame from BRG to HSV
    hsv_img = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    cv2.imshow('hsv1_img',hsv_img)
    cv2.waitKey(100)

    #Extract colored contours
    frame_threshed = cv2.inRange(hsv_img, COLOR_MIN, COLOR_MAX)
    thresh= frame_threshed.copy()

    cv2.imshow('gray_converted',thresh)
    cv2.waitKey(20)
    cv2.destroyAllWindows()

    contours,h = cv2.findContours(thresh,1,2)
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt,0.02*cv2.arcLength(cnt,True),True)
        if cv2.contourArea(cnt)<100 or not(cv2.isContourConvex(approx)):
            #print "contour is not convex    ", len(approx)
            continue
        if len(approx)==3:
            print "triangle"
            cv2.drawContours(frame,[cnt],0,(0,255,0),-1)
            labelDetectedObject(cnt, frame,"triangle")
        elif len(approx)==5:
            print "pentagon"
            cv2.drawContours(frame,[cnt],0,255,-1)
            labelDetectedObject(cnt, frame,"pentagon")
        elif len(approx)==4:
            print "square"
            cv2.drawContours(frame,[cnt],0,(0,0,255),-1)
            labelDetectedObject(cnt, frame,"square")
#        elif len(approx) == 9:
#            print "half-circle"
#            cv2.drawContours(frame,[cnt],0,(255,255,0),-1)
        elif len(approx) > 15:
            print "circle"
            cv2.drawContours(frame,[cnt],0,(0,255,255),-1)
            labelDetectedObject(cnt, frame,"circle")
        else:
#            circles = cv2.HoughCircles(thresh,cv2.HOUGH_GRADIENT,1,20,param1=50,param2=30,minRadius=0,maxRadius=0)
#            circles = np.uint16(np.around(circles))
#            for i in circles[0,:]:
#                # draw the outer circle
#                cv2.circle(frame,(i[0],i[1]),i[2],(0,255,0),2)
#                # draw the center of the circle
#                cv2.circle(frame,(i[0],i[1]),2,(0,0,255),3)
            centers = []
            radii = []


            br = cv2.boundingRect(cnt)
            radii.append(br[2])

            m = cv2.moments(cnt)
            center = (int(m['m10'] / m['m00']), int(m['m01'] / m['m00']))
            centers.append(center)

            print("There are {} circles".format(len(centers)))

            radius = int(np.average(radii)) + 5

            for center in centers:
#                cv2.circle(frame, center, 3, (255, 0, 0), -1)
#                cv2.circle(frame, center, radius, (0, 255, 0), 1)
                cv2.drawContours(frame,[cnt],0,(0,255,255),-1)
                labelDetectedObject(cnt, frame,"circle")


    cv2.imshow('thresh',thresh)
    cv2.waitKey(100)
    cv2.destroyAllWindows()
    return frame



def labelDetectedObject(cnt, frame, text):
    object_rect=cv2.minAreaRect(cnt)
    box = cv2.cv.BoxPoints(object_rect)
    box = np.int0(box)
    minX=min(box[0][0],box[1][0],box[2][0],box[3][0])
    minY=min(box[0][1],box[1][1],box[2][1],box[3][1])
    maxX=max(box[0][0],box[1][0],box[2][0],box[3][0])
    maxY=max(box[0][1],box[1][1],box[2][1],box[3][1])
    center = (math.fabs(minX+(maxX-minX)/2.0), math.fabs(minY+(maxY-minY)/2.0))
    cv2.putText(frame,text, (int(center[0]), int(center[1])), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (255,255,255),1)




#function  takes an image and returns the bounded rectangle with detected color shades
def detectColor(image, COLOR_MIN,COLOR_MAX):
    #Convert to PIL image
    original=convert2pil(image)
    print "original ",type(original)

    #convert to numpy array
    frame=np.asarray(original[:,:])
    print "frame ",type(frame)

    # cv2.imshow('frame',frame)
    # cv2.waitKey()

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, COLOR_MIN, COLOR_MAX)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask)

    #to control what parts of conture are selected
    # cv2.imshow('frame',frame)
    # cv2.waitKey()



    #convert frame from BRG to HSV
    hsv_img = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    cv2.imshow('hsv_img',hsv_img)
    cv2.waitKey(100)
    cv2.destroyAllWindows()
    #Extract colored contours
    frame_threshed = cv2.inRange(hsv_img, COLOR_MIN, COLOR_MAX)
    thresh= frame_threshed.copy()



    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)


    #find colored spot of biggest area
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt=contours[max_index]

    #make a bounded rectangle around this spot
    #object_rect=cv2.boundingRect(cnt)
    #(134, 350, 26, 54)

    #another way to retrieve coordinates of vertices of rotated rectangle:
    #first get the rectangle with min area

    object_rect=cv2.minAreaRect(cnt)
    #((504.2386474609375, 343.97100830078125), (213.28689575195312, 274.3262023925781), -3.7112834453582764)
    print "Rectangle", object_rect

    return object_rect


if __name__ == '__main__':
    IP = "10.104.4.40"
    PORT = 9559
    naoImage = getColour(IP, PORT)

