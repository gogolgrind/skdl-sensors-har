#!/usr/bin/env python

from events import Events
from queue import Queue
import threading
import time

class ModuleProto(object):

    #this is a Class Prototype
    #do not override or modify anything here
    def __init__(self, *args):
        self.d_ev = Events(('output'))
        self.rd_buff = Queue(1)
        self.output = self.d_ev.output
        self.event_id = 0

        self.c_ev = Events(('output'))
        self.rc_buff = Queue(1)
        self.coutput = self.c_ev.output

        self.init(*args)

    def input(self, ev):
        self.rd_buff.put(ev)

    def cinput(self, ev):
        self.rc_buff.put(ev)

    def input_nowait(self, ev):
        try: self.rd_buff.put_nowait(ev)
        except: pass
        

    def cinput_nowait(self, ev):
        try: self.rc_buff.put_nowait(ev)
        except: pass

    def start(self):
        self.t = threading.Thread(target=self.update, name=self.thread_name, args=())
        self.t.daemon = True
        self.t.start()

        self.cthread_name = self.thread_name + ".c"
        self.ct = threading.Thread(target=self.cupdate, name=(self.cthread_name), args=())
        self.ct.daemon = True
        self.ct.start()
        return self

    def isAlive(self):
        return(self.t.isAlive())

    def get(self):
        ev = self.rd_buff.get()
        return ev

    def cget(self):
        ev = self.rc_buff.get()
        return ev

    def send(self, *args):
        self.event_id += 1
        self.output(args)

    #do not override!
    def csend(self, *args):
        self.coutput(args)
    
    def set_buff_size(self, x):
        self.rd_buff = Queue(x)
    
    def set_cbuff_size(self, x):
        self.rc_buff = Queue(x)

    def get_buff_size(self):
        return(self.rd_buff.maxsize)

    def get_cbuff_size(self):
        return(self.rc_buff.maxsize)

    #override in the Subclass instead of the default __init__
    #preserve param format: init(self, *args)
    def init(self, *args):
        self.thread_name = "EventsProto"

    def cupdate(self):
        while self.isAlive():
            self.csend(self.thread_name, self.event_id)
            time.sleep(1)
        self.csend(self.thread_name, None)
