# This test demonstrates how to use the ALPhotoCapture module.
# # Note that you might not have this module depending on your distribution
# import os
# import sys
# import time
# from naoqi import ALProxy
# import cv2
# import math
#
#
# cwd = os.getcwd()
# print cwd
#
#




import sys
import time
import cv2
import math

# Python Image Library
import Image

from naoqi import ALProxy
from naoqi import ALBroker
from naoqi import ALModule

IP = "10.104.4.40"
PORT = 9559

"""
First get an image from Nao, then show it on the screen with PIL.
"""
#path = "/home/nao/images/"
#path = "/home/guenthse/uni/semesterprojekt/nao_images/"
path = "/home/vahan/PycharmProjects/visionForLegoGame"

myBroker = ALBroker("myBroker",
                    "0.0.0.0", # listen to anyone
                     0, # find a free port and use it
                     IP, # parent broker IP
                     PORT) # parent broker port

camProxy = ALProxy("ALVideoDevice", IP, PORT)
resolution = 2 # VGA
colorSpace = 11 # RGB

# xValue = x
# yValue = y

#for b in range(0, 5):
areas = [0,0,0]
colors = ['red', 'green', 'blue']

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

# Create a PIL Image from our pixel array.
im = Image.fromstring("RGB", (imageWidth, imageHeight), array)

# Save the image.
im.save(path+ "/camImage" + str(t0) + ".jpg", "JPEG")

im.show()

