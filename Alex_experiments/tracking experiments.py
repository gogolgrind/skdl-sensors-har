#!/usr/bin/env python

# Python 2/3 compatibility
from __future__ import print_function
from imutils.video import WebcamVideoStream
import numpy as np
import cv2 as cv
import imutils

import matplotlib.pyplot as plt

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

import time


import os#, pickle
def run_in_separate_process(func, *args, **kwds):
    pread, pwrite = os.pipe()
    pid = os.fork()
    if pid > 0:
        os.close(pwrite)
        with os.fdopen(pread, 'rb') as f:
            status, result = cPickle.load(f)
        os.waitpid(pid, 0)
        if status == 0:
            return result
        else:
            raise result
    else: 
        os.close(pread)
        try:
            result = func(*args, **kwds)
            status = 0
        except ValueError:
            result = ValueError
            status = 1
        #with os.fdopen(pwrite, 'wb') as f:
        #    try:
        #        cPickle.dump((status,result), f, cPickle.HIGHEST_PROTOCOL)
        #    except (cPickle.PicklingError, exc):
        #        cPickle.dump((2,exc), f, cPickle.HIGHEST_PROTOCOL)
        os._exit(0)

pose_size = (656,368)
#pose_size = (432,368)

def resize(frame, forward = True, mode = "opt"):


    if(mode=="opt"):
        pct = 1
        newsize = (int(frame.shape[1] * pct), int(frame.shape[0] * pct))
        #frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    elif(mode=="pose"):
        newsize = pose_size



    if (forward):
        frame = cv.resize(frame, newsize, interpolation = cv.INTER_AREA)
    else:
        frame = cv.resize(frame, oldsize, interpolation = cv.INTER_AREA)

    return(frame)

def show_fps(frame, fps):
    frame = cv.putText(frame,
                    "FPS: %f" % (fps),
                    (10, 10),  cv.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
    return(frame)


def draw_points(frame, points):
    for point in points[0]:
        point = (int(point[0]*frame.shape[:2][1]), int(point[1]*frame.shape[:2][0]))
        cv.circle(frame, point, 10, (0, 0, 255), thickness=3, lineType=8, shift=0)

    return frame

def get_pose_points(humans):
    centers = {}
    for human_idx in range(len(humans)):
        for i in humans[human_idx].body_parts:
            body_part = humans[human_idx].body_parts[i]
            center = (body_part.x, body_part.y)
            centers[(human_idx,i)] = center
    return centers

def filter_frame(frame):
    
    #frame = cv.flip(frame,1)
    
    #clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    #for i in range(0,3):   
        #frame[:,:,i] = clahe.apply(frame[:,:,i])

    #frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    return(frame)


def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv.remap(img, flow, None, cv.INTER_NEAREST)
    return res

def set_new_points(p0, tr_points, reused, frame_limiter):

    if tr_points and p0:
        p_merged = {}
        for key in tr_points.keys():
            #tr_keys[key] = key
            if key not in p_merged:
                p_merged[key] = 0
        for key in p0.keys():
            #p0_keys[key] = key
            if key not in p_merged:
                p_merged[key] = 0
    elif tr_points:
        return(tr_points, {})
    else:
        return(p0, {})
    
    for key in p_merged:
        if (key in tr_points.keys()) and (key in p0.keys()):
            tr_points[key] = tr_points[key]# + (p0[key] - tr_points[key])/5
            #if key not in reused:
            #    reused[key] = 0
            #reused[key] = reused[key] + 1

        if key not in tr_points.keys():
            if key not in reused:
                reused[key] = 0
            reused[key] = reused[key]+1
            tr_points[key] = p0[key]
        if reused and key in reused and reused[key]>frame_limiter:
            del reused[key]
            del tr_points[key]
                
    return(tr_points, reused)

if __name__ == '__main__':
    import sys
    print(__doc__)
    print(cv.getBuildInformation())
    import pafy

    #url = 'https://www.youtube.com/watch?v=czWf1vbNwoQ'
    #url = 'https://www.youtube.com/watch?v=xctzp0dp9uc'
    #url = 'https://www.youtube.com/watch?v=diVpN-ofRwQ'
    #url = 'https://www.youtube.com/watch?v=iwN2r8Lma_8'
    #url = 'https://www.youtube.com/watch?v=RyiLIIx2sEI'
    #url = 'https://www.youtube.com/watch?v=BfsRFyxDDoA'
    #url = 'https://www.youtube.com/watch?v=L_GMj-ymMSY'
    #url = 'https://www.youtube.com/watch?v=Ax7tmaYObIQ'
    #url = 'https://www.youtube.com/watch?v=4b8cK9QwSJE'
    #url = 'https://www.youtube.com/watch?v=oHbeguUPmVE'
    #url = 'https://www.youtube.com/watch?v=fhx7iXkiLXQ'
    #url = 'https://www.youtube.com/watch?v=dmaJGcxN4O0'
    #url = 'https://www.youtube.com/watch?v=mFZuquf2kbs'
    #vPafy = pafy.new(url)
    #play = vPafy.getbest(preftype="webm")
    #wc_src = play.url

    #wc_src = "/dev/video0"
    wc_src = "http://192.168.43.1:8080/video"
    cam = cv.VideoCapture(wc_src)# WebcamVideoStream(src=wc_src).start()
    #wc_src = "http://192.168.43.1:8080/video"
    #wc_src = 0
    #cam = WebcamVideoStream(src=wc_src).start()


    
    model = ["mobilenet_thin", pose_size]
    e = TfPoseEstimator(get_graph_path(model[0]), target_size=model[1])
    fps_time = 0
    p0 = []
    p1 = []
    
    ret, prev = cam.read()
    
    oldsize = (int(prev.shape[1]*2), int(prev.shape[0]*2))
    prevgray = resize(filter_frame(prev), mode = "opt")

    tr_points = []
    frames_skiped = 0
    pose_reinit = False
    reused = {}
    fps_avg = []

    while True:
        ret, gray = cam.read()
        gray = filter_frame(gray)

        gray_opt = resize(gray, mode = "opt")
        gray_pose = resize(gray, mode = "pose")
        
        if p0:
            p0_trans = np.array([list(p0.values())], np.float32)
            p1, error, _ = cv.calcOpticalFlowPyrLK(prevgray, gray_opt, \
                        p0_trans, None, **dict(winSize = (28,28),
                 maxLevel = 12222,
                 criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30000, 0.0)
                 ,minEigThreshold = 1e-2
                 ) )
            #p1_rev, error, _ = cv.calcOpticalFlowPyrLK(gray_opt, prevgray, \
            #            p1, None, **lk_params)
            #p1_fov, error, _ = cv.calcOpticalFlowPyrLK(prevgray, gray_opt, \
            #            p1_rev, None, **lk_params)

            if(len(p1[0])==len(p0_trans[0])):
                count = 0
                for key in p0:
                    p0[key] = p1[0][count]
                    count = count+1
            else:
                pose_reinit = True
        else:
            pose_reinit = True

        if frames_skiped >= 5:
            pose_reinit = True      

        prevgray = gray_opt

        ch = 0xFF & cv.waitKey(1)
        if ch == 27:
            break
        if ch == ord('1') or pose_reinit:
            humans = e.inference(gray_pose, True, 2)
                     #run_in_separate_process(e.inference,[e,gray_pose, False, 2])
            tr_points_temp = tr_points
            tr_points = get_pose_points(humans)
            if tr_points:
                for tr_idx in tr_points:
                    tr_points[tr_idx] = np.array(tr_points[tr_idx])*tuple(reversed(gray_opt.shape[:2]))          
                    #p0 = tr_points

            else:
                tr_points = tr_points_temp
            
            frames_skiped = 0
            pose_reinit = False
           
       


         
        (p0, reused) = set_new_points(p0, tr_points, reused, 0)

            

        gray_pose = resize(gray_pose, False, mode = "pose")


        fps = 1.0 / (time.time() - fps_time)
        fps_avg.append(fps)
        fps_time = time.time()
        if(len(fps_avg)>200):
            del fps_avg[0]

        if p0:
            p1_vis = np.array([list(p0.values())], np.float32)
            for p_idx in range(0,len(p1_vis[0])):
                p1_vis[0][p_idx] = p1_vis[0][p_idx]/tuple(reversed(gray_opt.shape[:2]))
            gray_pose = draw_points(gray_pose, p1_vis)
        cv.imshow('flow HSV', show_fps(gray_pose, sum(fps_avg)/200))
        frames_skiped = frames_skiped + 1
        #time.sleep(0.1)
        
        
        



cv.destroyAllWindows()
