#!/usr/bin/env python
from ..ModuleProto import ModuleProto
import cv2 as cv

class RawVideo(ModuleProto):

    def init(self, *args):
        flow_name = args[0]
        src = args[1]
        url = args[2]
        if(bool(url)):
            import pafy
            vPafy = pafy.new(src)
            play = vPafy.getbest(preftype="webm")
            src = play.url
        
        self.stream = cv.VideoCapture(src)
        self.thread_name = flow_name + ".Raw"
 
    def update(self):
        while True:
            (grabbed, frame) = self.stream.read()
            if(grabbed):
                self.send(self.thread_name, frame)
            else:
                self.send(self.thread_name, None)
                break
