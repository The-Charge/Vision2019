import libjevois as jevois
import cv2
import numpy as np

class Chessboard:
    
    def __init__(self):
    
        self.CHESSBOARD_HEIGHT = 7
        self.CHESSBOARD_WIDTH = 5
        self.TILE_LENGTH = 1.125
        
        self.NUM_COEFFS = 5

        # termination criteria
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        self.objp = np.zeros((self.CHESSBOARD_WIDTH*self.CHESSBOARD_HEIGHT,3), np.float32)
        self.objp[:,:2] = np.mgrid[0:self.CHESSBOARD_HEIGHT:self.TILE_LENGTH,0:self.CHESSBOARD_WIDTH:self.TILE_LENGTH].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        self.objpoints = [] # 3d point in real world space
        self.imgpoints = [] # 2d points in image plane.
        
        self.dist_data = [[], [], [], [], []]

        self.frame = 0
    
    def processNoUSB(self, inframe):
        img = inframe.getCvBGR()
        
        output, dist_coeffs = self.find_distortion(img)
        toSend = str(dist_coeffs) + '\n'
        jevois.sendSerial(toSend)
        
        self.frame += 1
        
    def process(self, inframe, outframe):
        img = inframe.getCvBGR()
        
        output, dist_coeffs = self.find_distortion(img)
        toSend = str(dist_coeffs) + '\n'
        jevois.sendSerial(toSend)
        
        outframe.sendCv(output)
        
        self.frame += 1
    
    def find_distortion(self, img):
        # Prepare list for return value
        avg_dist = [[], [], [], [], []]
        
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (self.CHESSBOARD_HEIGHT, self.CHESSBOARD_WIDTH),None)
        
            
        # If found, add object points, image points (after refining them)
        if ret == True:
            self.objpoints = [self.objp]

            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),self.criteria)
            self.imgpoints = [corners2]

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (self.CHESSBOARD_HEIGHT,self.CHESSBOARD_WIDTH), corners2,ret)
            
            # Returns camera matrix, distortion coefficients, rotation and translation vectors
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, gray.shape[::-1],None,None)
        
            # Add the constants to the overall list
            for i in range(self.NUM_COEFFS):
                self.dist_data[i].append(dist[0][i])
            
            # Calculate new averages
            for i, data in enumerate(self.dist_data):
                avg = np.mean(data)
                std = np.std(data)
                avg_dist[i] = (avg, std)
        
        return img, avg_dist