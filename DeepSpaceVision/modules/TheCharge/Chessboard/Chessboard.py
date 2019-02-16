import libjevois as jevois
import cv2
import numpy as np

class Chessboard:
    
    def __init__(self):
        self.frame = 0
    
    def processNoUSB(self, inframe):
        img = inframe.getCvBGR()
        cv2.imwrite('/jevois/modules/TheCharge/Chessboard/img{}.png'.format(self.frame), img)
        jevois.sendSerial('Saved file img{}.png to storage'.format(self.frame))
        self.frame += 1
        
    def process(self, inframe, outframe):
        img = inframe.getCvBGR()
        cv2.imwrite('/jevois/modules/TheCharge/Chessboard/img{}.png'.format(self.frame), img)
        jevois.sendSerial('Saved file img{}.png to storage'.format(self.frame))
        self.frame += 1
        
        outframe.sendCv(img)