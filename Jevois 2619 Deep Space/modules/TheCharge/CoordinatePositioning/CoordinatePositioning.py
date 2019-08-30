import libjevois as jevois
import cv2
import numpy as np

TARGET_STRIP_WIDTH = 2.0             # inches
TARGET_STRIP_LENGTH = 5.5            # inches
TARGET_STRIP_CORNER_OFFSET = 4.0     # inches
TARGET_STRIP_ROT = math.radians(14.5)

cos_a = math.cos(TARGET_STRIP_ROT)
sin_a = math.sin(TARGET_STRIP_ROT)

pt = [TARGET_STRIP_CORNER_OFFSET, 0.0, 0.0]
right_strip = [tuple(pt), ]  # this makes a copy, so we are safe
pt[0] += TARGET_STRIP_WIDTH * cos_a
pt[1] += TARGET_STRIP_WIDTH * sin_a
right_strip.append(tuple(pt))
pt[0] += TARGET_STRIP_LENGTH * sin_a
pt[1] -= TARGET_STRIP_LENGTH * cos_a
right_strip.append(tuple(pt))
pt[0] -= TARGET_STRIP_WIDTH * cos_a
pt[1] -= TARGET_STRIP_WIDTH * sin_a
right_strip.append(tuple(pt))

# left strip is mirror of right strip
left_strip = [(-p[0], p[1], p[2]) for p in right_strip]



class CoordinatePositioning:
    
    def __init__(self):
        self.outside_target_coords = numpy.array([self.left_strip[2], self.left_strip[1],
                                                  self.right_strip[1], self.right_strip[2]])
        '''
        [[-7.3133853  -4.82405201  0.        ]
         [-5.93629528  0.50076001  0.        ]
         [ 5.93629528  0.50076001  0.        ]
         [ 7.3133853  -4.82405201  0.        ]]
        '''
        print(self.outside_target_coords)
    
    
    def processNoUSB(self, inframe):
        inimg = inframe.getCvBGR()
        
        
    #def process(self, inframe, outframe)
    #    pass