import libjevois as jevois
import cv2
import numpy as np
import time
import math

## Incorporating GRIP Pipeline into Jevois code, using templates from Anand Rajamani
#
# Add some description of your module here.
#
# @author Nathaniel Kalantar
# 
# @videomapping YUYV 640 480 30 YUYV 640 480 30 TheCharge DeepSpaceVision
# @email 
# @address 123 first street, Los Angeles CA 90012, USA
# @copyright Copyright (C) 2018 by Nathaniel Kalantar
# @mainurl 
# @supporturl 
# @otherurl 
# @license 
# @distribution Unrestricted
# @restrictions None
# @ingroup modules

class DeepSpaceVision:
    # ###################################################################################################
    ## Constructor
    def __init__(self):
        # Boolean
        self.EDIT_IMAGE = True
    
        # 3D Reconstruction constants
        self.obj_points = [(-4.0, 0.0, 0.0),
                      (-5.936295280756216, 0.5007600081088829, 0.0),
                      (-7.313385303055644, -4.82405201397071, 0.0),
                      (-5.377090022299429, -5.324812022079593, 0.0),
                      
                      (4.0, 0.0, 0.0),
                      (5.936295280756216, 0.5007600081088829, 0.0),
                      (7.313385303055644, -4.82405201397071, 0.0),
                      (5.377090022299429, -5.324812022079593, 0.0)]
                      
        self.cam_matrix = [[251.14969233879845, 0.0, 160],
                      [0.0, 258.8135140705428, 120],
                      [0.0, 0.0, 1.0]]
                      
        self.dist_coeff = [1.1082291807813722,
                           -64.992034623241494,
                           -0.023849353550161684,
                           -0.022550316575808114,
                           1954.505249457445]
                           
        # Distortion values currently not working: use empty list
        self.dist_coeff = []
        
        # Change into numpy float32 type arrays
        self.obj_points = np.float32(self.obj_points)
        self.cam_matrix = np.float32(self.cam_matrix)
        self.dist_coeff = np.float32(self.dist_coeff)
        
        # Instantiate a JeVois Timer to measure our processing framerate:
        self.timer = jevois.Timer("processing timer", 100, jevois.LOG_INFO)
        
        # a simple frame counter used to demonstrate sendSerial():
        self.frame = 0
        
        # GRIP constants
        self.__blur_radius = 0.6289311175076467

        self.blur_output = None

        self.__cv_extractchannel_src = self.blur_output
        self.__cv_extractchannel_channel = 1.0

        self.cv_extractchannel_output = None

        self.__cv_threshold_src = self.cv_extractchannel_output
        self.__cv_threshold_thresh = 30.0
        self.__cv_threshold_maxval = 255.0
        self.__cv_threshold_type = cv2.THRESH_BINARY

        self.cv_threshold_output = None

        self.__mask_input = self.blur_output
        self.__mask_mask = self.cv_threshold_output

        self.mask_output = None

        self.__normalize_input = self.mask_output
        self.__normalize_type = cv2.NORM_MINMAX
        self.__normalize_alpha = 0.0
        self.__normalize_beta = 255.0

        self.normalize_output = None

        self.__hsv_threshold_input = self.normalize_output
        self.__hsv_threshold_hue = [64.10701515877801, 80.53474711861844]
        self.__hsv_threshold_saturation = [220.0, 255.0]
        self.__hsv_threshold_value = [75.41505716401612, 255.0]

        self.hsv_threshold_output = None

        self.__cv_erode_src = self.hsv_threshold_output
        self.__cv_erode_kernel = None
        self.__cv_erode_anchor = (-1, -1)
        self.__cv_erode_iterations = 2.0
        self.__cv_erode_bordertype = cv2.BORDER_CONSTANT
        self.__cv_erode_bordervalue = (-1)

        self.cv_erode_output = None

        self.__cv_dilate_src = self.cv_erode_output
        self.__cv_dilate_kernel = None
        self.__cv_dilate_anchor = (-1, -1)
        self.__cv_dilate_iterations = 1.0
        self.__cv_dilate_bordertype = cv2.BORDER_CONSTANT
        self.__cv_dilate_bordervalue = (-1)

        self.cv_dilate_output = None

        self.__find_contours_input = self.cv_dilate_output
        self.__find_contours_external_only = True

        self.find_contours_output = None

        self.__filter_contours_contours = self.find_contours_output
        self.__filter_contours_min_area = 15.0
        self.__filter_contours_min_perimeter = 0.0
        self.__filter_contours_min_width = 0.0
        self.__filter_contours_max_width = 1000.0
        self.__filter_contours_min_height = 0.0
        self.__filter_contours_max_height = 1000.0
        self.__filter_contours_solidity = [0.0, 100]
        self.__filter_contours_max_vertices = 1000000.0
        self.__filter_contours_min_vertices = 5.0
        self.__filter_contours_min_ratio = 0.1
        self.__filter_contours_max_ratio = 0.9

        self.filter_contours_output = None
        
    # ###################################################################################################
    ## Process function with no USB output
    def processNoUSB(self, inframe):
        # Get the next camera image (may block until it is captured) and here convert it to OpenCV BGR. If you need a
        # grayscale image, just use getCvGRAY() instead of getCvBGR(). Also supported are getCvRGB() and getCvRGBA():
        inimg = inframe.getCvBGR()

        # Start measuring image processing time (NOTE: does not account for input conversion time):
        self.timer.start()
        
        # Get frames/s info from our timer
        fps = self.timer.stop()
        
        self.findTargets(inimg)

        self.frame += 1
        
    # ###################################################################################################
    ## Process function with USB output
    def process(self, inframe, outframe):
        # Get the next camera image (may block until it is captured) and here convert it to OpenCV BGR by default. If
        # you need a grayscale image instead, just use getCvGRAY() instead of getCvBGR(). Also supported are getCvRGB()
        # and getCvRGBA():
        inimg = inframe.getCvBGR()
        outimg = inframe.getCvBGR()
        
        # Start measuring image processing time (NOTE: does not account for input conversion time):
        self.timer.start()
        
        outimg = self.findTargets(inimg)
        
        # Convert our output image to video output format and send to host over USB:
        outframe.sendCv(outimg)
        
        self.frame += 1
    
    # ###################################################################################################
    ## Process an image and processes the contours
    def findTargets(self, img):
        #GRIP CODE
        
        # Step Blur0:
        self.__blur_input = img
        (self.blur_output) = self.__blur(self.__blur_input, self.__blur_radius)

        # Step CV_extractChannel0:
        self.__cv_extractchannel_src = self.blur_output
        (self.cv_extractchannel_output) = self.__cv_extractchannel(self.__cv_extractchannel_src, self.__cv_extractchannel_channel)

        # Step CV_Threshold0:
        self.__cv_threshold_src = self.cv_extractchannel_output
        (self.cv_threshold_output) = self.__cv_threshold(self.__cv_threshold_src, self.__cv_threshold_thresh, self.__cv_threshold_maxval, self.__cv_threshold_type)

        # Step Mask0:
        self.__mask_input = self.blur_output
        self.__mask_mask = self.cv_threshold_output
        (self.mask_output) = self.__mask(self.__mask_input, self.__mask_mask)

        # Step Normalize0:
        self.__normalize_input = self.mask_output
        (self.normalize_output) = self.__normalize(self.__normalize_input, self.__normalize_type, self.__normalize_alpha, self.__normalize_beta)

        # Step HSV_Threshold0:
        self.__hsv_threshold_input = self.normalize_output
        (self.hsv_threshold_output) = self.__hsv_threshold(self.__hsv_threshold_input, self.__hsv_threshold_hue, self.__hsv_threshold_saturation, self.__hsv_threshold_value)

        # Step CV_erode0:
        self.__cv_erode_src = self.hsv_threshold_output
        (self.cv_erode_output) = self.__cv_erode(self.__cv_erode_src, self.__cv_erode_kernel, self.__cv_erode_anchor, self.__cv_erode_iterations, self.__cv_erode_bordertype, self.__cv_erode_bordervalue)

        # Step CV_dilate0:
        self.__cv_dilate_src = self.cv_erode_output
        (self.cv_dilate_output) = self.__cv_dilate(self.__cv_dilate_src, self.__cv_dilate_kernel, self.__cv_dilate_anchor, self.__cv_dilate_iterations, self.__cv_dilate_bordertype, self.__cv_dilate_bordervalue)

        # Step Find_Contours0:
        self.__find_contours_input = self.cv_dilate_output
        (self.find_contours_output) = self.__find_contours(self.__find_contours_input, self.__find_contours_external_only)

        # Step Filter_Contours0:
        self.__filter_contours_contours = self.find_contours_output
        (self.filter_contours_output) = self.__filter_contours(self.__filter_contours_contours, self.__filter_contours_min_area, self.__filter_contours_min_perimeter, self.__filter_contours_min_width, self.__filter_contours_max_width, self.__filter_contours_min_height, self.__filter_contours_max_height, self.__filter_contours_solidity, self.__filter_contours_max_vertices, self.__filter_contours_min_vertices, self.__filter_contours_min_ratio, self.__filter_contours_max_ratio)
        
        #END GRIP CODE
        
        # Sort contors from left to right on the screen
        contours = sorted(self.filter_contours_output, key = lambda x: get_contour_extreme_points(x)[0])
        
        # Iterate through contours and store some of their values
        contourInfo = []
        
        for cnt in contours:
            
            # Coordinates of center of contour
            cx, cy = get_contour_coords(cnt)
            
            # Width and height
            left, right, top, bottom = get_contour_extreme_points(cnt)
            width = right[0]-left[0]
            height = bottom[1]-top[1]
            
            # Contour's angle of rotation in range [-90, 90]
            angle = get_contour_angle(cnt)
            
            # Store the contour information in a new list in order (contour, center, shape, angle)
            contourInfo.append((cnt, (cx, cy), (width, height), angle))
            
            if self.EDIT_IMAGE:
                # Draw contour
                cv2.drawContours(img, [cnt], 0, (255, 0, 0), 2)
                
                # Draw a circle around the center of the contour
                cv2.circle(img, (cx, cy), 1, (0, 0, 0), 2)
                
                # Draw a line through the target
                p1, p2 = line_through_contour(cnt)
                cv2.line(img, p1, p2, (255, 255, 255), 1)
        
        # Information to send over serial
        toSend = ""
        
        # Find target pairs
        for i in range(len(contourInfo)-1):
            
            cnt_l, (cx_l, cy_l), (width_l, height_l), angle_l = contourInfo[i]
            cnt_r, (cx_r, cy_r), (width_r, height_r), angle_r = contourInfo[i+1]
            
            # If the targets are pointed in the correct orientation: / \
            if angle_l > 0 and angle_r < 0 and cx_l < cx_r:
                img_points = []
                    
                l_left, l_right, l_top, l_bottom = get_contour_extreme_points(cnt_l)
                img_points.append(l_right)
                img_points.append(l_top)
                img_points.append(l_left)
                img_points.append(l_bottom)
                
                r_left, r_right, r_top, r_bottom = get_contour_extreme_points(cnt_r)
                img_points.append(r_left)
                img_points.append(r_top)
                img_points.append(r_right)
                img_points.append(r_bottom)
                
                img_points = np.float32(img_points)
                ret, rvec, tvec = cv2.solvePnP(self.obj_points, img_points, self.cam_matrix, self.dist_coeff)
                
                if ret:
                    distance, yaw, rotation = compute_output_values(rvec, tvec)
                    c_x = int(np.mean([cx_l, cx_r]))
                    c_y = int(np.mean([cy_l, cy_r]))
                    # Yaw calculation from matrices is not as accurate, so it's recalculated here
                    yaw = math.degrees(math.atan( (160 - c_x) / 251.1496923 ))
                    
                    # Append the target coordinates to the serial information
                    if toSend != "":
                        toSend = toSend + ","
                    toSend += '{:.4},{:.4},{:.4}'.format(distance, yaw, rotation)
                    
                    if self.EDIT_IMAGE:           
                        cv2.putText(img, "{:.4}\"".format(distance), (c_x, c_y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255))
                        cv2.putText(img, "{:.4}\"".format(yaw), (c_x, c_y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255))
                        cv2.putText(img, "{:.4}\"".format(rotation), (c_x, c_y+40), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255))
                
                if self.EDIT_IMAGE:
                    # Draw a box around both contours
                    rx, ry, rw, rh = rect = box_contours([cnt_l, cnt_r])
                    cv2.rectangle(img, rect, (0, 255, 255), 2)
                    
                    # Find the center of the two targets
                    tar_x, tar_y = (int(rx+rw/2), int(ry+rh/2))
                    
                    # Draw a circle in the middle of both contours
                    cv2.circle(img, (tar_x, tar_y), 2, (255, 255, 255), 4)
            
        # Don't send an empty string
        if toSend != "":
            jevois.sendSerial(toSend)
        
        return img
        
    # OpenCV functions generated by GRIP
    @staticmethod
    def __blur(src, radius):
        """Softens an image using one of several filters.
        Args:
            src: The source mat (numpy.ndarray).
            radius: The radius for the blur as a float.
        Returns:
            A numpy.ndarray that has been blurred.
        """
        ksize = int(2 * round(radius) + 1)
        return cv2.blur(src, (ksize, ksize))

    @staticmethod
    def __cv_extractchannel(src, channel):
        """Extracts given channel from an image.
        Args:
            src: A numpy.ndarray.
            channel: Zero indexed channel number to extract.
        Returns:
             The result as a numpy.ndarray.
        """
        return cv2.extractChannel(src, (int) (channel + 0.5))

    @staticmethod
    def __cv_threshold(src, thresh, max_val, type):
        """Apply a fixed-level threshold to each array element in an image
        Args:
            src: A numpy.ndarray.
            thresh: Threshold value.
            max_val: Maximum value for THRES_BINARY and THRES_BINARY_INV.
            type: Opencv enum.
        Returns:
            A black and white numpy.ndarray.
        """
        return cv2.threshold(src, thresh, max_val, type)[1]

    @staticmethod
    def __mask(input, mask):
        """Filter out an area of an image using a binary mask.
        Args:
            input: A three channel numpy.ndarray.
            mask: A black and white numpy.ndarray.
        Returns:
            A three channel numpy.ndarray.
        """
        return cv2.bitwise_and(input, input, mask=mask)

    @staticmethod
    def __normalize(input, type, a, b):
        """Normalizes or remaps the values of pixels in an image.
        Args:
            input: A numpy.ndarray.
            type: Opencv enum.
            a: The minimum value.
            b: The maximum value.
        Returns:
            A numpy.ndarray of the same type as the input.
        """
        return cv2.normalize(input, None, a, b, type)

    @staticmethod
    def __hsv_threshold(input, hue, sat, val):
        """Segment an image based on hue, saturation, and value ranges.
        Args:
            input: A BGR numpy.ndarray.
            hue: A list of two numbers the are the min and max hue.
            sat: A list of two numbers the are the min and max saturation.
            lum: A list of two numbers the are the min and max value.
        Returns:
            A black and white numpy.ndarray.
        """
        out = cv2.cvtColor(input, cv2.COLOR_BGR2HSV)
        return cv2.inRange(out, (hue[0], sat[0], val[0]),  (hue[1], sat[1], val[1]))

    @staticmethod
    def __cv_erode(src, kernel, anchor, iterations, border_type, border_value):
        """Expands area of lower value in an image.
        Args:
           src: A numpy.ndarray.
           kernel: The kernel for erosion. A numpy.ndarray.
           iterations: the number of times to erode.
           border_type: Opencv enum that represents a border type.
           border_value: value to be used for a constant border.
        Returns:
            A numpy.ndarray after erosion.
        """
        return cv2.erode(src, kernel, anchor, iterations = (int) (iterations +0.5),
                            borderType = border_type, borderValue = border_value)

    @staticmethod
    def __cv_dilate(src, kernel, anchor, iterations, border_type, border_value):
        """Expands area of higher value in an image.
        Args:
           src: A numpy.ndarray.
           kernel: The kernel for dilation. A numpy.ndarray.
           iterations: the number of times to dilate.
           border_type: Opencv enum that represents a border type.
           border_value: value to be used for a constant border.
        Returns:
            A numpy.ndarray after dilation.
        """
        return cv2.dilate(src, kernel, anchor, iterations = (int) (iterations +0.5),
                            borderType = border_type, borderValue = border_value)

    @staticmethod
    def __find_contours(input, external_only):
        """Sets the values of pixels in a binary image to their distance to the nearest black pixel.
        Args:
            input: A numpy.ndarray.
            external_only: A boolean. If true only external contours are found.
        Return:
            A list of numpy.ndarray where each one represents a contour.
        """
        if(external_only):
            mode = cv2.RETR_EXTERNAL
        else:
            mode = cv2.RETR_LIST
        method = cv2.CHAIN_APPROX_SIMPLE
        contours, hierarchy = cv2.findContours(input, mode=mode, method=method)
        return contours

    @staticmethod
    def __filter_contours(input_contours, min_area, min_perimeter, min_width, max_width,
                        min_height, max_height, solidity, max_vertex_count, min_vertex_count,
                        min_ratio, max_ratio):
        """Filters out contours that do not meet certain criteria.
        Args:
            input_contours: Contours as a list of numpy.ndarray.
            min_area: The minimum area of a contour that will be kept.
            min_perimeter: The minimum perimeter of a contour that will be kept.
            min_width: Minimum width of a contour.
            max_width: MaxWidth maximum width.
            min_height: Minimum height.
            max_height: Maximimum height.
            solidity: The minimum and maximum solidity of a contour.
            min_vertex_count: Minimum vertex Count of the contours.
            max_vertex_count: Maximum vertex Count.
            min_ratio: Minimum ratio of width to height.
            max_ratio: Maximum ratio of width to height.
        Returns:
            Contours as a list of numpy.ndarray.
        """
        output = []
        for contour in input_contours:
            x,y,w,h = cv2.boundingRect(contour)
            if (w < min_width or w > max_width):
                continue
            if (h < min_height or h > max_height):
                continue
            area = cv2.contourArea(contour)
            if (area < min_area):
                continue
            if (cv2.arcLength(contour, True) < min_perimeter):
                continue
            hull = cv2.convexHull(contour)
            solid = 100 * area / cv2.contourArea(hull)
            if (solid < solidity[0] or solid > solidity[1]):
                continue
            if (len(contour) < min_vertex_count or len(contour) > max_vertex_count):
                continue
            ratio = (float)(w) / h
            if (ratio < min_ratio or ratio > max_ratio):
                continue
            output.append(contour)
        return output
    
# Gets the area of the contour
def getArea(con): 
    return cv2.contourArea(con)
    
# Finds the center coordinates of a contour
def get_contour_coords(cnt):
    M = cv2.moments(cnt)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = 0, 0
    return cx, cy
    
# Find the leftmost, rightmost, topmost, and bottommost points of a contour
def get_contour_extreme_points(cnt):
    leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
    rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
    topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
    bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])
    
    return leftmost, rightmost, topmost, bottommost
    
def get_contour_angle(cnt):
    try:
        # Fit an ellipse to the contour and find the angle
        ellipse = (x, y), (MA, ma), angle = cv2.fitEllipse(cnt)
        
        if angle > 90:  # Maps the ellipse angle to [-90, 90] for trig functions
            angle = angle-180
        
        return angle
    except:
        return 0
    
# Returns two points to draw a line through a target
def line_through_contour(cnt):
    cx, cy = get_contour_coords(cnt)
    angle = get_contour_angle(cnt)
    
    px1 = int(cx + 400*math.cos(math.radians(90-angle)))
    py1 = int(cy - 400*math.sin(math.radians(90-angle)))
    
    px2 = int(cx - 400*math.cos(math.radians(90-angle)))
    py2 = int(cy + 400*math.sin(math.radians(90-angle)))
    
    return (px1, py1), (px2, py2)
    
# Returns an array representing an angled rectangle around a contour
def angled_rectangle(cnt):
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    
    return box

# Returns a rectangle that contains all contours in the list
def box_contours(cnt_list):
    leftmost = 319
    rightmost = 0
    topmost = 239
    bottommost = 0
    
    for cnt in cnt_list:
        left, right, top, bottom = get_contour_extreme_points(cnt)
        
        if left[0] < leftmost:
            leftmost = left[0]
        if right[0] > rightmost:
            rightmost = right[0]
        if top[1] < topmost:
            topmost = top[1]
        if bottom[1] > bottommost:
            bottommost = bottom[1]
    
    return (leftmost, topmost, rightmost-leftmost, bottommost-topmost)
    
def compute_output_values(rvec, tvec):
        '''Compute the necessary output distance and angles'''

        # The tilt angle only affects the distance and angle1 calcs

        x = tvec[0][0]
        z = math.sin(0.0) * tvec[1][0] + math.cos(0.0) * tvec[2][0]

        # distance in the horizontal plane between camera and target
        distance = math.sqrt(x**2 + z**2)

        # horizontal angle between camera center line and target
        angle1 = math.degrees(math.atan2(x, z))

        rot, _ = cv2.Rodrigues(rvec)
        rot_inv = rot.transpose()
        pzero_world = matmul(rot_inv, -tvec)
        angle2 = math.degrees(math.atan2(pzero_world[0][0], pzero_world[2][0]))
        #angle2 = 0.0

        return distance, angle1, angle2

def matmul(a, b):
    '''Multiply two matrices and return the output as a list'''
    
    try:
        zip_b = zip(*b)
        zip_b = list(zip_b)
        return [[sum(ele_a*ele_b for ele_a, ele_b in zip(row_a, col_b)) 
                 for col_b in zip_b] for row_a in a]
    except:
        return [[]]