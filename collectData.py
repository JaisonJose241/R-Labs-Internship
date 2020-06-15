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
count = {}
flag = 0
createVideo = 0

    
file = {48 : 0,
        49 : 1,
        50 : 2,
        51 : 3,
        52 : 4,
        53 : 5,
        54 : 6}        
    
if not os.path.exists("data"):
        os.makedirs("data")
        os.makedirs("data/TRAIN")
        os.makedirs("data/TEST")
        
        os.makedirs("data/TRAIN/0")
        os.makedirs("data/TRAIN/1")
        os.makedirs("data/TRAIN/2")
        os.makedirs("data/TRAIN/3")
        os.makedirs("data/TRAIN/4")
        os.makedirs("data/TRAIN/5")
        
        os.makedirs("data/TEST/0")
        os.makedirs("data/TEST/1")
        os.makedirs("data/TEST/2")
        os.makedirs("data/TEST/3")
        os.makedirs("data/TEST/4")
        os.makedirs("data/TEST/5")


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
    cv2.putText(blank_screen, 'NO. OF IMAGES :   TRAIN  ||  TEST', (20, 46), font, 1, color[1], 1)
    
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
    
    cv2.putText(blank_screen, str(count['TRAIN'][file[48]]),  (190, 65), font, 1, color[0], 1)
    cv2.putText(blank_screen, str(count['TRAIN'][file[49]]),   (190, 85), font, 1, color[0], 1)
    cv2.putText(blank_screen, str(count['TRAIN'][file[50]]),   (190, 105), font, 1, color[0], 1)
    cv2.putText(blank_screen, str(count['TRAIN'][file[51]]), (190, 125), font, 1, color[0], 1)
    cv2.putText(blank_screen, str(count['TRAIN'][file[52]]),  (190, 145), font, 1, color[0], 1)
    cv2.putText(blank_screen, str(count['TRAIN'][file[53]]),  (190, 165), font, 1, color[0], 1)
    
    cv2.putText(blank_screen, str(count['TEST'][file[48]]),   (270, 65), font, 1, color[0], 1)
    cv2.putText(blank_screen, str(count['TEST'][file[49]]),    (270, 85), font, 1, color[0], 1)
    cv2.putText(blank_screen, str(count['TEST'][file[50]]),    (270, 105), font, 1, color[0], 1)
    cv2.putText(blank_screen, str(count['TEST'][file[51]]),  (270, 125), font, 1, color[0], 1)
    cv2.putText(blank_screen, str(count['TEST'][file[52]]),   (270, 145), font, 1, color[0], 1)
    cv2.putText(blank_screen, str(count['TEST'][file[53]]),   (270, 165), font, 1, color[0], 1)
    
    cv2.putText(blank_screen, 'EXIT -->  KEY : ESC', (305, 185), font, 1, color[1], 1)
    cv2.putText(blank_screen, 'STOP recording KEY: q   |', (10, 185), font, 1, color[1], 1)
    cv2.imshow('Info', blank_screen)
    
    
def checkLength():
    global count
    
    count = {
             'TRAIN'    :  {0:  len(os.listdir("data/TRAIN/0")),
                            1:  len(os.listdir("data/TRAIN/1")),
                            2:  len(os.listdir("data/TRAIN/2")),
                            3:  len(os.listdir("data/TRAIN/3")),
                            4:  len(os.listdir("data/TRAIN/4")),
                            5:  len(os.listdir("data/TRAIN/5"))},

             'TEST'     : {0:   len(os.listdir("data/TEST/0")),
                           1:   len(os.listdir("data/TEST/1")),
                           2:   len(os.listdir("data/TEST/2")),
                           3:   len(os.listdir("data/TEST/3")),
                           4:   len(os.listdir("data/TEST/4")),
                           5:   len(os.listdir("data/TEST/5"))} 
             }

'''
...............................................................MAIN FUNCTION.........................................
'''  
  
if __name__ == "__main__":
    mode = 'TRAIN'
        
    while True:
        ret, image = camera.read()
        diffImg = returnDifference()
        
        num_frames += 1        
        
        checkLength()
        blankScreenDisplay()
                
        if diffImg is not None:
            cv2.imshow('differenceImage', diffImg)
            diffImg = cv2.resize(diffImg, (128, 128))
        
        keyPressed = cv2.waitKey(1) & 0xFF                                       #ESC key to terminate and 0xFF for 64-bit computers            
        if flag == 1:
            cv2.imwrite('data/'+mode+'/'+str(file[createVideo])+'/'+str(count[mode][file[createVideo]])+'.jpg', diffImg)
            cv2.putText(image, 'RECORDING',   (10, 20), font, 2, color[0], 2)    
        if keyPressed == 27:
            break
        elif keyPressed == ord('m'):
            if mode == 'TEST':
                mode = 'TRAIN'
            else:
                mode='TEST'
            keyPressed = 255            
        elif keyPressed == ord('q'):
            flag = 0
            createVideo = 0            
        elif keyPressed is not None and keyPressed is not 255:
            flag = 1
            createVideo = keyPressed
        else:
            pass 

        cv2.imshow('original Image', image)
          
    camera.release()
    cv2.destroyAllWindows()    
        
            
    