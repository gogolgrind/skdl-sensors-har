#!/usr/bin/env python
from ..ModuleProto import ModuleProto
import cv2 as cv
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh


class PoseVideoBase(ModuleProto):
    
    def init(self, *args):
        flow_name = args[0]
        self.pose_size = (432,368)
        self.e = TfPoseEstimator(get_graph_path("mobilenet_thin"), target_size=self.pose_size)

        self.thread_name = flow_name + ".PoseBase"

    def update(self):
        while True:
            ev = self.get()
            frame = ev[1]
            if(frame is not None):
                frame = self.pose_estimate(frame)
                self.send(self.thread_name, frame)
            else:
                self.send(self.thread_name, None)
                break

    def resize(self, frame, old_size = None):

        if (old_size is None):
            frame = cv.resize(frame, self.pose_size, interpolation = cv.INTER_AREA)
        else:
            frame = cv.resize(frame, old_size, interpolation = cv.INTER_AREA)
        return(frame)
    
    def pose_estimate(self, frame): 
        
        old_size = (frame.shape[1], frame.shape[0])

        frame = self.resize(frame)

        humans,h2,h3,h4 = self.e.inference(frame, False, 2)
        frame = TfPoseEstimator.draw_humans(frame, humans, imgcopy=False)

        frame = self.resize(frame, old_size)
        return(frame)
