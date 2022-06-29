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
from sensor_msgs.msg import Image
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
        linear_velocity_descriptor = ParameterDescriptor(
            type=ParameterType.PARAMETER_DOUBLE,
            description='Linear velocity in m/s.')

        camera_image_topic_descriptor = ParameterDescriptor(
            type=ParameterType.PARAMETER_STRING,
            description='Camera image topic.')
        
        debug_image_topic_descriptor = ParameterDescriptor(
            type=ParameterType.PARAMETER_STRING,
            description='Run in debug mode and publish to debug image topic')

        command_topic_descriptor = ParameterDescriptor(
            type=ParameterType.PARAMETER_STRING,
            description='Twist command topic name output.')

        debug_descriptor = ParameterDescriptor(
            type=ParameterType.PARAMETER_BOOL,
            description='Debug boolean.')
        
        self.declare_parameter("linear_velocity", 1.0, 
            linear_velocity_descriptor)
        
        self.declare_parameter("camera_image", "/NPU/image_sim", 
            camera_image_topic_descriptor)
        
        self.declare_parameter("debug_image", "/CupVisionDebug", 
            debug_image_topic_descriptor)

        self.declare_parameter("cmd_topic", "/cmd_vel", 
            command_topic_descriptor)

        self.declare_parameter("debug", False, 
            debug_descriptor)

        self.cameraImageTopic = self.get_parameter("camera_image").value

        self.debugImageTopic = self.get_parameter("debug_image").value

        self.commandTopic = self.get_parameter("cmd_topic").value

        self.debug = self.get_parameter("debug").value

        self.linearVelocity = self.get_parameter("linear_velocity").value


        #setup CvBridge
        self.bridge = CvBridge()
        
        self.timeStamp = self.get_clock().now().nanoseconds
        
        #Subscribers
        self.imageSub = self.create_subscription(Image, 
            '{:s}'.format(self.cameraImageTopic), 
            self.ImageCallback, 
            qos_profile_sensor_data)

        #Publishers
        if self.debug:
            self.debugDetectionImagePub = self.create_publisher(Image, self.debugImageTopic, 10)

        self.cmd = Twist()
        self.cmdPub = self.create_publisher(Twist, '{:s}'.format(self.commandTopic), 10)

        self.pts1 = np.float32([[0, 88], [300, 88], [750, 167], [-450, 167]])
        self.pts2 = np.float32([[0, 0], [300, 0], [300, 300], [0, 300]])

        self.matrix = cv2.getPerspectiveTransform(self.pts1, self.pts2)

    def findLines(self, passedImage):
        
        self.timeStamp = self.get_clock().now().nanoseconds    
        
        passedImageHSV = cv2.cvtColor(passedImage,cv2.COLOR_RGB2HSV)
        lowerYellow = np.array([70,140,180])
        upperYellow = np.array([120,200,220])
        passedImageHSVThresh = cv2.inRange(passedImageHSV, lowerYellow, upperYellow)

        #Find contours
        cnts, hierarchy = cv2.findContours(passedImageHSVThresh.copy(),
            cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)


        #Max number of found contours to process based on area of return, largest returned first
        maxCnt = 2
        if len (cnts) < maxCnt:
            maxCnt = len(cnts)
        cnt = sorted(cnts, key=cv2.contourArea, reverse=True)[0:maxCnt]

        vectorArray = []

        #Loop through contours
        for cn in cnt:

            if self.debug:
                #Paint all the areas found in the contour
                cv2.fillPoly(passedImage,pts=[cn],color=(0,0,255))
            
            #Find lines from contours using least square method
            vectorArray.append(cv2.fitLine(cn,cv2.DIST_L2,0,0.01,0.01))

        try:
            avgVector = (vectorArray[0] + vectorArray[1]) / 2
        except IndexError:
            try: 
                avgVector = vectorArray[0]
            except IndexError:
                avgVector = [-1, -1, -1, -1]

        vx,vy,x,y = avgVector

        normalizedVectorSlope = vx/vy
        if(normalizedVectorSlope > 1 or normalizedVectorSlope < -1):
            normalizedVectorSlope = 0.0
        distanceFromCenter = (x - 150) / 150

        self.linearVelocity = 1.25

        angularVelocity = (normalizedVectorSlope*.5) + (-1*distanceFromCenter*.5)

        self.cmd.linear.x = self.linearVelocity
        self.cmd.angular.z = float(angularVelocity)
        self.cmdPub.publish(self.cmd)

        return passedImage
      
    
    def ImageCallback(self, data):
        
        # Scene from subscription callback
        scene = self.bridge.imgmsg_to_cv2(data, "bgr8")

        sceneWarp = cv2.warpPerspective(scene, self.matrix, (300, 300), cv2.BORDER_CONSTANT, 0)

        #find lines function
        sceneDetected = self.findLines(sceneWarp)
        
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
