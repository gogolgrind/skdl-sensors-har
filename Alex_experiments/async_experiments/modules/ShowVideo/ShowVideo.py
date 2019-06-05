#!/usr/bin/env python
from ..ModuleProto import ModuleProto
import cv2 as cv

class ShowVideo(ModuleProto):
    def init(self, *args):
        self.displayList = list()
        self.thread_name = "ShowThread"

    def update(self):
        while True:
            self.show_frame(self.get())

    def show_frame(self, x):
        window_name = x[0]
        frame = x[1]
        
        self.start_window(window_name)

        if(frame is None):
            self.close_window(window_name)
        else:
            cv.imshow(window_name, frame)
            cv.waitKey(1) #critical      

    def start_window(self, window_name):
        if(window_name not in self.displayList):
            self.displayList.append(window_name)
            cv.namedWindow(window_name)

    def close_window(self, window_name):
        self.displayList.remove(window_name)
        cv.destroyWindow(window_name)
