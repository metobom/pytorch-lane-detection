#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import torch
from torchvision import transforms
import cv2
from torch_model import network, w_b_init
import time
from displayers import show_weighted
import imageio
import os
import torchvision
from detect_pts import detect

import rospy
from cv_bridge import CvBridge, CvBridgeError
import sys
from sensor_msgs.msg import Image
import numpy as np


class detector():
    def __init__(self):
        # DETECTOR
        self.model = torch.load('saved_models/lane_model_final.pth')
        self.model.eval()
        self.transform = transforms.ToTensor()
        self.predict_size = (416, 240)
        self.device = device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.color_filter = cv2.imread('green.png')
        # ROS
        self.node_name = 'detector'
        rospy.init_node(self.node_name)
        self.loader = transforms.Compose([transforms.ToTensor()])
        self.bridge = CvBridge()  
        rospy.Subscriber("/mybot/camera1/image_raw", Image, self.ros_to_cv, queue_size = 1)
        rospy.loginfo("Image alındı.")

    def predict_image(self, image, model):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (self.predict_size))
        w, h = image.shape[1], image.shape[0]
        image = image.reshape(1, h, w)

        batch = torch.tensor(image / 255).unsqueeze(0).float()
        with torch.no_grad():
            batch = batch.to(self.device)
            output = model(batch) 
        output = output.cpu().numpy()
        return output

    def predict_ros(self, cv_image):
        t0 = time.time()
        cv_image = cv2.resize(cv_image, self.predict_size)
        w, h = self.predict_size
        out = self.predict_image(cv_image, self.model) * 255
        out = out.reshape(h, w, 1)
        out = np.array(out, dtype = np.uint8)
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
        '''
        pts = detect(out)
        for i in pts:
            u, v = i.ravel()
            out = cv2.circle(out, (u, v), 3, (0, 0, 255))
        
        black_out = out
    
        black_out = cv2.bitwise_and(black_out, self.color_filter)
        out = show_weighted(cv_image, 0.7, black_out, 0.5)
        '''    

        predict_time = time.time() - t0
        FPS = 1 / predict_time
        cv2.putText(out, 'FPS: {}'.format(str(FPS)), (w - int(w * 0.9), h - int(h * 0.9)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255))
        cv2.putText(out, 'Quality: {}p'.format(str(cv_image.shape[0])), (w - int(w * 0.9) + 30, h - int(h * 0.9) + 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255))
        return out
    
    def ros_to_cv(self, ros_image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(ros_image, 'bgr8')
        except CvBridgeError as error:
            print(error)       

        output = self.predict_ros(cv_image)

        cv2.imshow('output', output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
           rospy.signal_shutdown('Kapanıyor')
    

def main(args):
    try:
        detector()
        rospy.spin()
    except KeyboardInterrupt:
        print('Kapanıyor')
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)