import libjevois as jevois
import cv2
import numpy as np

class DriverCamCompression:
    
    def __init__(self):
        self.timer = jevois.Timer("processing timer", 100, jevois.LOG_INFO)
        self.frame = 0
        
    
    def processNoUSB(self, inframe):
        inimg = inframe.getCvBGR()
        self.timer.start()
        self.frame += 1
        
    
    def process(self, inframe, outframe):
        outimg = inimg = inframe.getCvBGR()
        self.timer.start()
        self.frame += 1
        
        
        
        outframe.sendCv(outimg)