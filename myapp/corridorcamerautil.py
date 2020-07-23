# -*- coding: utf-8 -*-

import tensorflow as tf
import threading
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import cv2
import os
import time
import subprocess
import argparse

# 控制陌生人检测
fall_timing = 0 # 计时开始
fall_start_time = 0 # 开始时间
fall_limit_time = 1 # if >= 1 seconds, then he/she falls.

# 全局变量
model_path = 'myapp/final/models/fall_detection.hdf5'
output_fall_path = 'myapp/final/supervision/fall'
 # your python path
python_path = '~/anaconda3/envs/tensorflow/bin/python'

# 全局常量
TARGET_WIDTH = 64
TARGET_HEIGHT = 64

# 初始化摄像头
#if not input_video:
#   vs = cv2.VideoCapture(0)
#   time.sleep(2)
#else:
#   vs = cv2.VideoCapture(input_video)

# 加载模型
model = load_model(model_path)
graph = tf.get_default_graph()
    
print('[INFO] 开始检测是否有人摔倒...')


class RecordingThread (threading.Thread):
    def __init__(self, name, camera, save_video_path):
        threading.Thread.__init__(self)
        self.name = name
        self.isRunning = True

        self.graph=graph
        self.cap = camera
        fourcc = cv2.VideoWriter_fourcc(*'XVID') # MJPG
        self.out = cv2.VideoWriter(save_video_path, fourcc, 20.0, 
                                   (640,480), True)

    def run(self):
        global fall_timing
        global fall_start_time
        while self.isRunning:

            print("后台程序")
            ret, frame = self.cap.read()
            if ret:
                print("corrior")
                frame = cv2.flip(frame, 1)
                self.out.write(frame)

                roi = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
                roi = roi.astype("float") / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                # determine facial expression
                with self.graph.as_default():
                    (fall, normal) = model.predict(roi)[0]
                label = "Fall (%.2f)" % (fall) if fall > normal else "Normal (%.2f)" % (normal)

                # display the label and bounding box rectangle on the output frame
                cv2.putText(frame, label, (frame.shape[1] - 150, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

                if fall > normal:
                    if fall_timing == 0:  # just start timing
                        fall_timing = 1
                        fall_start_time = time.time()
                    else:  # alredy started timing
                        fall_end_time = time.time()
                        difference = fall_end_time - fall_start_time

                        current_time = time.strftime('%Y-%m-%d %H:%M:%S',
                                                     time.localtime(time.time()))

                        if difference < fall_limit_time:
                            print('[INFO] %s, 走廊, 摔倒仅出现 %.1f 秒. 忽略.'
                                  % (current_time, difference))
                        else:  # strangers appear
                            event_desc = '有人摔倒!!!'
                            event_location = '走廊'
                            print('[EVENT] %s, 走廊, 有人摔倒!!!' % (current_time))
                            cv2.imwrite(os.path.join(output_fall_path,
                                                     'snapshot_%s.jpg'
                                                     % (time.strftime('%Y%m%d_%H%M%S'))), frame)
                            # insert into database
                            command = '%s myapp/final/inserting.py --event_desc %s --event_type 3 --event_location %s' % (
                            python_path, event_desc, event_location)
                            p = subprocess.Popen(command, shell=True)

        self.out.release()

    def stop(self):
        self.isRunning = False

    def __del__(self):
        self.out.release()

class CorridorCamera(object):
    def __init__(self):
        # Open a camera
        self.cap = cv2.VideoCapture(0)
        self.graph=graph
        # Initialize video recording environment
        self.is_record = False
        self.out = None
        
        # Thread for recording
        self.recordingThread = None
    
    def __del__(self):
        self.cap.release()
    
    def get_frame(self):
        global fall_timing
        global fall_start_time
        ret, frame = self.cap.read()

        if ret:
            frame = cv2.flip(frame, 1)
            roi = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
                
            # determine facial expression
            with self.graph.as_default():
                (fall, normal) = model.predict(roi)[0]
            label = "Fall (%.2f)" %(fall) if fall > normal else "Normal (%.2f)" %(normal)
            
            # display the label and bounding box rectangle on the output frame
            cv2.putText(frame, label, (frame.shape[1] - 150, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            
            if fall > normal:
                if fall_timing == 0: # just start timing
                    fall_timing = 1
                    fall_start_time = time.time()
                else: # alredy started timing
                    fall_end_time = time.time()
                    difference = fall_end_time - fall_start_time
                        
                    current_time = time.strftime('%Y-%m-%d %H:%M:%S',
                                                 time.localtime(time.time()))
                    
                    if difference < fall_limit_time:
                        print('[INFO] %s, 走廊, 摔倒仅出现 %.1f 秒. 忽略.' 
                                                 %(current_time, difference))
                    else: # strangers appear
                        event_desc = '有人摔倒!!!'
                        event_location = '走廊'
                        print('[EVENT] %s, 走廊, 有人摔倒!!!' %(current_time))
                        cv2.imwrite(os.path.join(output_fall_path, 
                                                 'snapshot_%s.jpg' 
                                    %(time.strftime('%Y%m%d_%H%M%S'))), frame)
                        # insert into database
                        command = '%s myapp/final/inserting.py --event_desc %s --event_type 3 --event_location %s' %(python_path, event_desc, event_location)
                        p = subprocess.Popen(command, shell=True)  

            ret, jpeg = cv2.imencode('.jpg', frame)

            return jpeg.tobytes()
      
        else:
            return None

    def start_record(self, save_video_path):
        self.is_record = True
        self.recordingThread = RecordingThread(
                               "Video Recording Thread", 
                               self.cap, save_video_path)
        self.recordingThread.start()

    def stop_record(self):
        self.is_record = False

        if self.recordingThread != None:
            self.recordingThread.stop()
