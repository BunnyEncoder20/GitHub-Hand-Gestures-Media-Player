#------------------ Media Player Controller Using Hand Gestures------------------

#Step1 : Import Libraries 
#           -read the camera feed
#           -create the trackbar window for color adjustment
#           -Feed the video feed frame by frame for preprocessing 
#Step2 : convert the frames into hsv (hue saturation value)
#Step3 : track hand on color basis
#Step4 : Create mask on basis of color and filter actual color




#Step - 1 Import Libraries and other important stuff
import cv2                 #openCV
import numpy            #as np
import math
import pyautogui     #as p    #used to virtual press the     keyboard keys when gestured by hands. Basically simulates keyboard presses
import time               #as t

#Reading camera feed 
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)                                        #0 = primary camera, 1 = secondary (external) camera 

#creating the trackbars for adjustments 
def do_nothing(x):              
    pass                                                                                                     #function which does nothing for when none of the trackbars are being used, it's call this function which will do nothing 
                                                                                                                #done so that the values of the trackbar can be dynamically updated 
cv2.namedWindow("Color Adjustments", cv2.WINDOW_NORMAL)         #naming the window containing the trackbars 
cv2.resizeWindow("Color Adjustments", (300,300))
cv2.createTrackbar("Thresh","Color Adjustments", 0, 255, do_nothing)

#Color Detection Track , naming them and placing them on the window we named above
cv2.createTrackbar("Lower_H", "Color Adjustments", 0, 255, do_nothing)              #specifying the ranges
cv2.createTrackbar("Lower_S", "Color Adjustments", 0, 255, do_nothing)
cv2.createTrackbar("Lower_V", "Color Adjustments", 0, 255, do_nothing)

cv2.createTrackbar("Upper_H", "Color Adjustments", 255, 255, do_nothing)
cv2.createTrackbar("Upper_S", "Color Adjustments", 255, 255, do_nothing)
cv2.createTrackbar("Upper_V", "Color Adjustments", 255, 255, do_nothing)


while True:                                                             
    _,frame = cam.read()                                                                            #reading frame by frame from the ablve cam object which is reading the entire video feed
    frame = cv2.flip(frame,2)                                                                     #frames are flipped to avoid mirror effect of webcams
    frame = cv2.resize(frame,(600,500))                                                       #resize to fit multple windows on the screen

    #Reading the Hand gesture data from the rectangle (blue) sub window
    cv2.rectangle(frame,(0,1), (300,500), (255,0,0),0)              
    crop_image = frame[1:500, 0:300]                                                           #defines the area from where the Hand Gestures will be read



    #Step - 2 - convert the frames into hsv (hue saturation value)
    hsv = cv2.cvtColor(crop_image, cv2.COLOR_BGR2HSV)



    #Step - 3 - tracking the hand on color basis
    l_h = cv2.getTrackbarPos("Lower_H", "Color Adjustments")            #getting all the values of the 6 trackbars from the window "Color Adjustment"
    l_s = cv2.getTrackbarPos("Lower_S", "Color Adjustments")
    l_v = cv2.getTrackbarPos("Lower_V", "Color Adjustments")

    u_h = cv2.getTrackbarPos("Upper_H", "Color Adjustments")
    u_s = cv2.getTrackbarPos("Upper_S", "Color Adjustments")
    u_v = cv2.getTrackbarPos("Upper_V", "Color Adjustments")

    lower_bound = numpy.array([l_h, l_s, l_v])             #note that the color hsv values can also be hardcoded here 
    upper_bound = numpy.array([u_h, u_s, u_v])          #but that will never be user friendly and hence we provide a GUI solution to that which is also dynamic



    #Step - 4 - creating mask and filtering actual color
    req_mask = cv2.inRange(hsv, lower_bound, upper_bound)                   #making the mask. Now only the colors which are specified by the range of upper and lower bound will be taken. Note it also takes the hsv image we created earlier
    filtr = cv2.bitwise_and(crop_image, crop_image, mask=req_mask)      #Using that mask here to filter using a bitwise add function



    #Till here we have gotten our webcam feed and filtered the colors, now to give it in representation



    #Step - 5 - Invert the Pixel Values and Enhancing results for better output
    mask1 = cv2.bitwise_not(req_mask)                                                               #inverting the pixels to be able to detect contours and convexcity defects - for this make the background black and the object which needs to be detected into white
    m_g = cv2.getTrackbarPos("Thresh", "Color Adjustments")                           
    ret, thresh = cv2.threshold(mask1, m_g, 255, cv2.THRESH_BINARY)               #thresh is used as a threshold value - values above it will be in white and below it will be in black. THRESH_BINARY converts the image into a single channel image (only black & white)
    dilate = cv2.dilate(thresh, (3,3), iterations = 6)                                              #used to reduce the noise significatently. If not then we will use erosion()



    #Step - 6 - Finally finding the contours here
    contours, hier = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #cv2.findContours(img, contour_retrival_mode, method)
    #contours can be seen as the red outline around the hand in the cam feed
    #now me can create convexcity hull and convexcity defects - to which we'll bind the hand guestures too 

    #if no contour is made/detected then this section might raise an error, hence the try block for avoiding a runtime error
    try:
        #Step - 7 - Find the contour with the maximum area
        contourmax = max(contours, key=lambda x: cv2.contourArea(x))
        epsilon = 0.0005*cv2.arcLength(contourmax,True)
        data = cv2.approxPolyDP(contourmax, epsilon, True)

        hull = cv2.convexHull(contourmax)
        #hull be seen as the green outline [blocky outline] around the detectd hand in the cam feed

        cv2.drawContours(crop_image, [contourmax], -1, (50,50,150), 2)
        cv2.drawContours(crop_image, [hull], -1, (0,255,0), 2)

        #Step - 8 - Finding the Comvexity Defects (from the convexity Hull)
        hull = cv2.convexHull(contourmax, returnPoints=False)
        defects = cv2.convexityDefects(contourmax, hull)                
        #convexityDefects finds the part of the mage which are blured or not properly captured
        #and stores it in a list named defects 

        count_defects = 0

        #print("Area==", cv2.countourArea(hull) - cv2.contourArea(contourmax))

        for i in range(defects.shape[0]):
            s,e,f,d = defects[i,0]

            #categorizing the results from the convexity defects
            start = tuple(contourmax[s][0])
            end = tuple(contourmax[e][0])
            far = tuple(contourmax[f][0])

            #Applying the Cosine rule

            a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
            angle = (math.acos((b ** 2 + c ** 2 - a ** 2) / (2 *  b * c)) * 180) / 3.14

            # if angel <= 50 draw a circle at the far point 
            # this shiz draws a the white point as seen in the hand outline 
            if angle <= 50 :
                count_defects += 1  #counting the number of defects detected
                cv2.circle(crop_image,far,5,[255,255,255],-1)
        
        

        #Step - 9 - Binding the hand guestures to the keyboard key presses
        #these are specific to a media player, Here we're using the VLC media player
        if count_defects == 0 :
            cv2.putText(frame," ",(50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,2),2)
            #when no hand is detected then do not show any text - doing nothing on the keyboard either 

        elif count_defects == 1:        #when a hand is detected by the camera
            pyautogui.press("space")    #space = play/pause
            cv2.putText(frame, "Play/Pause", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2)
            
        elif count_defects == 2 :       #when there are 3 fingers detected
            pyautogui.press("up") 
            cv2.putText(frame, "Volume Up", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2)
        
        elif count_defects == 3:        #when there are 4 fingers are detected
            pyautogui.press("down")
            cv2.putText(frame, "Volume Down", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2)

        elif count_defects == 4:        #when there are 5 finders detected 
            pyautogui.press("right")
            cv2.putText(frame, "Forward skip", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2)
        
        else:
            pass
    except:
        pass



    # Step - 10 - Showing all the results 
    cv2.imshow("Thresh" , thresh)
    #cv2.imshow("mask==",mask)
    cv2.imshow("filter",filtr)
    cv2.imshow("Result", frame)

    key = cv2.waitKey(25) &0xFF
    #to exit the program we use the esc key (key ASCII value = 27)
    if key == 27: 
        break

cam.release()
cv2.destroyAllWindows()


#Values Guide for detecting the hand :
#Lower_H = 90-95   ~ working for pastel dark blue background only for best results    
#Lower_S = 40-50
#Lower_V = 60-65