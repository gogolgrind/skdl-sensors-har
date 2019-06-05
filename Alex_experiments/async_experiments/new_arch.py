#!/usr/bin/env python
import time


from modules.MainFlow.MainFlow import MainFlow
  
mainFlow = MainFlow().start()

while mainFlow.isAlive():
    time.sleep(1)


