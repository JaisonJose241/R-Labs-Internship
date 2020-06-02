'''

startDate:  29 - 05 -2020
Project  :  Gesture Recognition
Part     :  Hand Detection with OpenCV Function writing
Libraries:  OpenCV (3.4.2),
            numpy (latest),
            os
Team ID  :  4

'''
'''
 ...........................................................Importing Required Libraries............................
'''
import cv2
import numpy as np
import os

'''
#............................................................Global Varibles........................................
'''

bckimg = None                                                                    #Background Image is set to none in begining
kernalOpen = np.ones((2, 2))                                                     #MORPH takes numpy array
kernalClose = np.ones((1, 1))                                                    #MORPH takes numpy array
(top, left), (bottom, right) = (250, 41), (419, 210)
num_frames = 0
camera = cv2.VideoCapture(0)
image = 0
font = cv2.FONT_HERSHEY_PLAIN
color = [(0, 255, 255), (255, 255, 0), (0, 255, 0)]                              #yellow, blue, green
mode = ''

trainCount = {'zero': len(os.listdir("data/train/0")),
             'one':   len(os.listdir("data/train/1")),
             'two':   len(os.listdir("data/train/2")),
             'three': len(os.listdir("data/train/3")),
             'four':  len(os.listdir("data/train/4")),
             'five':  len(os.listdir("data/train/5"))}

testCount = {'zero':  len(os.listdir("data/test/0")),
             'one':   len(os.listdir("data/test/1")),
             'two':   len(os.listdir("data/test/2")),
             'three': len(os.listdir("data/test/3")),
             'four':  len(os.listdir("data/test/4")),
             'five':  len(os.listdir("data/test/5"))}

'''
.............................................................FUNCTION  1............................................
Function Name : returnDifference(_)
Arguments     : Image from camera or input channel - img
About         : Flips horizontally the image, Draws Region of Interest BOX, 
                Average of 5 frames is taken, Finds difference from current frame, 
                Sets properties like thresholding and morphing
Return        : Substracted image after 5 frames - diff

ExtraPieceOf
Information   : Returns 'None' if no. of frames are less than 5                
'''
def returnDifference():                                                          # <------------- fUNCTION 1
    
    global image
    global bckimg
    image = cv2.resize(image, (420, 316))                                        # aspect ratio of 4:3
    image = cv2.flip(image, 1)
        
    cv2.rectangle(image, (top, left), (bottom, right), (0,255,0), 2)
                
    roi = image[left:right, top:bottom]
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi = cv2.GaussianBlur(roi, (1, 1), 0)
        
    if num_frames < 5:                                                       #taking average of 5 frames
        if bckimg is None:                                                   #for first frame
            bckimg = roi
        else:
            cv2.accumulateWeighted(roi, bckimg.astype("float"), 0.8)         #weighted average of pics only takes float values
        return None       
                   
    else:
        diff = cv2.absdiff(bckimg.astype("uint8"), roi)                      #takes difference with those averaged pics
        diff = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)[1]            #thresholding GRAY img
        diff = cv2.morphologyEx(diff, cv2.MORPH_OPEN, kernalOpen)            #MORPHING for ignoring small dots
        diff = cv2.morphologyEx(diff, cv2.MORPH_CLOSE, kernalClose)          #filling blank space in hand
        diff = cv2.resize(diff, (350, 350))
        return diff
    
def blankScreenDisplay():                                                        # <------------- fUNCTION 2
    global font
    global color
    global mode
    blank_screen = np.zeros((190, 500, 3), np.uint8)
    cv2.putText(blank_screen, 'MODE : ' + mode, (127, 30), font, 2, color[0], 2)
    cv2.putText(blank_screen, 'NO. OF IMAGES :   train  ||  test', (20, 46), font, 1, color[1], 1)
    
    cv2.putText(blank_screen, 'HI/BYE', (20, 65), font, 1, color[2], 1)
    cv2.putText(blank_screen, 'THUMBS UP', (20, 85), font, 1, color[2], 1)
    cv2.putText(blank_screen, 'PEACE/VICTORY', (20, 105), font, 1, color[2], 1)
    cv2.putText(blank_screen, 'NICE', (20, 125), font, 1, color[2], 1)
    cv2.putText(blank_screen, 'MY TURN', (20, 145), font, 1, color[2], 1)
    cv2.putText(blank_screen, 'HIT', (20, 165), font, 1, color[2], 1)
    
    
    cv2.putText(blank_screen, '(KEY : m)', (396, 30), font, 1, color[2], 1)
    cv2.putText(blank_screen, ':          |                    KEY : 0', (150, 65), font, 1, color[2], 1)
    cv2.putText(blank_screen, ':          |                    KEY : 1', (150, 85), font, 1, color[2], 1)
    cv2.putText(blank_screen, ':          |                    KEY : 2', (150, 105), font, 1, color[2], 1)
    cv2.putText(blank_screen, ':          |                    KEY : 3', (150, 125), font, 1, color[2], 1)
    cv2.putText(blank_screen, ':          |                    KEY : 4', (150, 145), font, 1, color[2], 1)
    cv2.putText(blank_screen, ':          |                    KEY : 5', (150, 165), font, 1, color[2], 1)
    
    cv2.putText(blank_screen, str(trainCount['zero']),  (200, 65), font, 1, color[0], 1)
    cv2.putText(blank_screen, str(trainCount['one']),   (200, 85), font, 1, color[0], 1)
    cv2.putText(blank_screen, str(trainCount['two']),   (200, 105), font, 1, color[0], 1)
    cv2.putText(blank_screen, str(trainCount['three']), (200, 125), font, 1, color[0], 1)
    cv2.putText(blank_screen, str(trainCount['four']),  (200, 145), font, 1, color[0], 1)
    cv2.putText(blank_screen, str(trainCount['five']),  (200, 165), font, 1, color[0], 1)
    
    cv2.putText(blank_screen, str(testCount['zero']),   (260, 65), font, 1, color[0], 1)
    cv2.putText(blank_screen, str(testCount['one']),    (260, 85), font, 1, color[0], 1)
    cv2.putText(blank_screen, str(testCount['two']),    (260, 105), font, 1, color[0], 1)
    cv2.putText(blank_screen, str(testCount['three']),  (260, 125), font, 1, color[0], 1)
    cv2.putText(blank_screen, str(testCount['four']),   (260, 145), font, 1, color[0], 1)
    cv2.putText(blank_screen, str(testCount['five']),   (260, 165), font, 1, color[0], 1)
    
    cv2.putText(blank_screen, 'EXIT -->  KEY : ESC', (305, 185), font, 1, color[1], 1)
    cv2.imshow('Info', blank_screen)
    
    
'''
...............................................................MAIN FUNCTION.........................................
'''    
if __name__ == "__main__":
    mode = 'TRAIN'
    # Create the directory structure
    if not os.path.exists("data"):
        os.makedirs("data")
        os.makedirs("data/train")
        os.makedirs("data/test")
        
        os.makedirs("data/train/0")
        os.makedirs("data/train/1")
        os.makedirs("data/train/2")
        os.makedirs("data/train/3")
        os.makedirs("data/train/4")
        os.makedirs("data/train/5")
        
        os.makedirs("data/test/0")
        os.makedirs("data/test/1")
        os.makedirs("data/test/2")
        os.makedirs("data/test/3")
        os.makedirs("data/test/4")
        os.makedirs("data/test/5")
    while True:
        ret, image = camera.read()
        diffImg = returnDifference()
        
        num_frames += 1        
        
        blankScreenDisplay()
        cv2.imshow('original Image', image)
        
        if diffImg is not None:
            cv2.imshow('differenceImage', diffImg)
        
        keyPressed = cv2.waitKey(1) & 0xFF                                       #ESC key to terminate and 0xFF for 64-bit computers
        if keyPressed == 27:
            break
        elif keyPressed == ord('m'):
            if mode == 'TEST':
                mode = 'TRAIN'
            else:
                mode='TEST'
        else:
            pass
          
    camera.release()
    cv2.destroyAllWindows()    
        
            
    