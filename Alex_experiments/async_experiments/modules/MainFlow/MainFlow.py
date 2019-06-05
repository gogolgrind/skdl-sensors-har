#!/usr/bin/env python
from ..ModuleProto import ModuleProto
import cv2 as cv

from ..RawVideo.RawVideo import RawVideo
from ..ShowVideo.ShowVideo import ShowVideo
from ..PoseVideoBase.PoseVideoBase import PoseVideoBase
from ..FaceLandmarkFast.FaceLandmarkFast import FaceLandmarkFast

class MainFlow(ModuleProto):

    def init(self, *args):
        self.thread_name = "MainThread.sys"
        self.thread_pool = {}
        self.set_cbuff_size(10)

    def get_thread_names(self, exclude = ".sys"):
        thr_list = list()
        for t in threading.enumerate():
            thr_name = t.getName()
            if (exclude not in thr_name):
                thr_list.append(thr_name)
        return(thr_list)

    def update(self):
        self.init_flow()
        while self.flow_running():
            self.sheduler()
        cv.destroyAllWindows()
    
    def flow_running(self):
        condition = self.thread_pool is not None
        return(condition)

    def sheduler(self):
        for i in range(0, self.get_cbuff_size()):
            x = self.cget()
            
            if(x[0] not in self.thread_pool):
                self.thread_pool[x[0]] = x[1]
            else:
                if(x[1] is not None):
                    self.thread_pool[x[0]] = x[1]
                else:
                    del(self.thread_pool[x[0]])
                    if(len(self.thread_pool) == 0):
                        self.thread_pool = None
        print(self.thread_pool)


    def init_flow(self):
        cam = RawVideo("cam1", 0, False)
        cam.coutput += self.cinput

        im_view = ShowVideo()
        
        #poses = [FaceLandmarkFast("cam2"), FaceLandmarkFast("cam2")]
        #for i in range(0,len(poses)):
        #    cam.output += poses[i].input_nowait
        #    poses[i].coutput += self.cinput
        #    poses[i].output += im_view.input_nowait
            

        #landmarks = [FaceLandmark("cam1")]
        
        #for i in range(0,len(landmarks)):
        #    poses[i].output += landmarks[i].input_nowait
        #    landmarks[i].coutput += self.cinput
        #    landmarks[i].output += im_view.input_nowait

        im_view.start()
        
        #for i in range(0,len(poses)):
        #    landmarks[i].start()
        #    poses[i].start()     

        cam.start()
