#!/usr/bin/env python3
import os
import sys
import copy
import re
import importlib
import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data
from rclpy.node import Node
from rclpy.exceptions import ParameterNotDeclaredException
from rcl_interfaces.msg import Parameter
from rcl_interfaces.msg import ParameterType
from rcl_interfaces.msg import ParameterDescriptor
import sensor_msgs.msg
from geometry_msgs.msg import Twist
import cv2
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile

if cv2.__version__ < "4.0.0":
    raise ImportError("Requires opencv >= 4.0, "
                      "but found {:s}".format(cv2.__version__))

class NXPTrackVision(Node):

    def __init__(self):

        super().__init__("nxp_track_vision")

        # Get paramaters or defaults
        pyramid_down_descriptor = ParameterDescriptor(
            type=ParameterType.PARAMETER_INTEGER,
            description='Number of times to pyramid image down.')

        camera_image_topic_descriptor = ParameterDescriptor(
            type=ParameterType.PARAMETER_STRING,
            description='Camera image topic.')
        
        debug_image_topic_descriptor = ParameterDescriptor(
            type=ParameterType.PARAMETER_STRING,
            description='Run in debug mode and publish to debug image topic')

        namespace_topic_descriptor = ParameterDescriptor(
            type=ParameterType.PARAMETER_STRING,
            description='Namespaceing if needed.')

        mask_ratio_array_descriptor = ParameterDescriptor(
            type=ParameterType.PARAMETER_DOUBLE_ARRAY,
            description='Array for mask ratio')
        
        self.declare_parameter("pyramid_down", 0, 
            pyramid_down_descriptor)
        
        self.declare_parameter("camera_image", "NPU", 
            camera_image_topic_descriptor)
        
        self.declare_parameter("debug_image", "", 
            debug_image_topic_descriptor)

        self.declare_parameter("namespace", "", 
            namespace_topic_descriptor)

        self.declare_parameter("mask_ratio_array", [1.0, 0.5], 
            mask_ratio_array_descriptor)

        self.pyrDown = self.get_parameter("pyramid_down").value

        self.cameraImageTopic = self.get_parameter("camera_image").value

        self.debugImageTopic = self.get_parameter("debug_image").value

        self.namespaceTopic = self.get_parameter("namespace").value

        self.mask_ratio_array = self.get_parameter("mask_ratio_array").value
        

        #setup CvBridge
        self.bridge = CvBridge()
        
        #Rectangualr area to remove from image calculation to 
        # eliminate the vehicle. Used as ratio of overall image width and height
        # "width ratio,height ratio"
        self.maskRectRatioWidthHeight = np.array([float(self.mask_ratio_array[0]),float(self.mask_ratio_array[1])])
        
        #Bool for generating and publishing the debug image evaluation
        self.debug = True

        self.timeStamp = self.get_clock().now().nanoseconds
        
        #Subscribers
        self.imageSub = self.create_subscription(sensor_msgs.msg.Image, 
            '/{:s}/image_sim'.format(self.cameraImageTopic), 
            self.pixyImageCallback, 
            qos_profile_sensor_data)

        #Publishers
        self.debugDetectionImagePub = self.create_publisher(sensor_msgs.msg.Image,
            '/DebugImage2', 10)

        self.cmdVel = Twist()
        self.cmdVelPub = self.create_publisher(Twist, '/requested_vel', 10)

    def findLines(self, passedImage):
        
        self.timeStamp = self.get_clock().now().nanoseconds    
        
        passedImageHSV = cv2.cvtColor(passedImage,cv2.COLOR_RGB2HSV)
        lowerYellow = np.array([70,140,180])
        upperYellow = np.array([120,200,220])
        passedImageHSVThresh = cv2.inRange(passedImageHSV, lowerYellow, upperYellow)

        imageHeight, imageWidth = passedImageHSV.shape[:2]

        
        #Create image mask background
        maskWhite = np.ones(passedImageHSV.shape[:2], dtype="uint8") * 255
        
        #calculate points to be masked based on provided ratio
        maskVehicleBoxTopLeftXY = (int(imageWidth*(1.0-self.maskRectRatioWidthHeight[0])/2.0), 
            int(imageHeight*(1.0-self.maskRectRatioWidthHeight[1])))
        
        #calculate points to be masked based on provided ratio
        maskVehicleBoxBottomRightXY = (int(imageWidth*(1.0+self.maskRectRatioWidthHeight[0])/2.0), 
            int(imageHeight))
        
        maskVehicle = cv2.rectangle(maskWhite,maskVehicleBoxTopLeftXY,
            maskVehicleBoxBottomRightXY,color=0,thickness=-1)
        
        #Find contours
        cnts, hierarchy = cv2.findContours(passedImageHSVThresh.copy(),
            cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        returnedImageDebug=passedImage

        #Max number of found contours to process based on area of return, largest returned first
        maxCnt = 2
        if len (cnts) < maxCnt:
            maxCnt = len(cnts)
        cnt = sorted(cnts, key=cv2.contourArea, reverse=True)[0:maxCnt]

        # #Take largest contours and sort left to right
        # if len (cnts) > 1:
        #     boundingBoxes = [cv2.boundingRect(c) for c in cnt]
        #     (cnt, boundingBoxes) = zip(*sorted(zip(cnt, boundingBoxes),
        #         key=lambda b:b[1][0], reverse=self.sortRightToLeft))

        #Initialize and/or clear existing found line vector array
        pixyScaledVectorArray = np.empty([0,4], int)

        #Used to determine what line is mapped in message from debug image
        lineNumber=0

        vectorArray = []

        #Loop through contours
        for cn in cnt:

            if self.debug:
                #Paint all the areas found in the contour
                cv2.fillPoly(returnedImageDebug,pts=[cn],color=(0,0,255))
            
            #Find lines from contours using least square method
            #[vectorX,vectorY,linePointX,linePointY] = cv2.fitLine(cn,cv2.DIST_L2,0,0.01,0.01)
            vectorArray.append(cv2.fitLine(cn,cv2.DIST_L2,0,0.01,0.01))

        try:
            avgVector = (vectorArray[0] + vectorArray[1]) / 2
        except IndexError:
            try: 
                avgVector = vectorArray[0]
            except IndexError:
                avgVector = [-1, -1, -1, -1]
        #print(avgVector)

        vx,vy,x,y = avgVector

        normalizedVectorSlope = vx/vy
        if(normalizedVectorSlope > 1 or normalizedVectorSlope < -1):
            normalizedVectorSlope = 0.0
        distanceFromCenter = (x - 150) / 150

        #print(vy)
        #print(vx)        
        #print(normalizedVectorSlope)
        #print(distanceFromCenter)

        linearVelocity = 1.25

        angularVelocity = (normalizedVectorSlope*.5) + (-1*distanceFromCenter*.5)
        #print("angularVelocity: " + str(angularVelocity))


        self.cmdVel.linear.x = linearVelocity
        self.cmdVel.angular.z = float(angularVelocity)
        self.cmdVelPub.publish(self.cmdVel)

        return returnedImageDebug
      
    
    def pixyImageCallback(self, data):
        
        # Scene from subscription callback
        scene = self.bridge.imgmsg_to_cv2(data, "bgr8")

        print(scene.shape)

        pts1 = np.float32([[0, 88], [300, 88], [750, 167], [-450, 167]])
        pts2 = np.float32([[0, 0], [300, 0], [300, 300], [0, 300]])

        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        scene = cv2.warpPerspective(scene, matrix, (300, 300), cv2.BORDER_CONSTANT, 0)

        #deep copy and pyramid down image to reduce resolution
        scenePyr = copy.deepcopy(scene)
        if self.pyrDown > 0:
            for i in range(self.pyrDown):
                scenePyr = cv2.pyrDown(scenePyr)
        sceneDetect = copy.deepcopy(scenePyr)

        #find lines function
        sceneDetected = self.findLines(sceneDetect)
        
        if self.debug:
            #publish debug image
            msg = self.bridge.cv2_to_imgmsg(sceneDetected, "bgr8")
            msg.header.stamp = data.header.stamp
            self.debugDetectionImagePub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = NXPTrackVision()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
