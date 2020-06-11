import numpy as np
from keras.models import model_from_json
import operator
import cv2
import sys, os

#....................................................................

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

#..............................................................

# Loading the model
json_file = open("model.json", "r")
model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

(top, left), (bottom, right) = (250, 41), (419, 210)
bckimg = None                                                                    #Background Image is set to none in begining
kernalOpen = np.ones((2, 2))                                                     #MORPH takes numpy array
kernalClose = np.ones((1, 1))                                                    #MORPH takes numpy array
num_frames = 0
camera = cv2.VideoCapture(0)
image = 0
font = cv2.FONT_HERSHEY_PLAIN
color = [(0, 255, 255), (255, 255, 0), (0, 255, 0)] 
#.............................................................................................

def returnDifference():                                                          # <------------- fUNCTION 1
    
    global image
    global bckimg
    global num_frames
    image = cv2.resize(image, (420, 316))                                        # aspect ratio of 4:3
    image = cv2.flip(image, 1)
        
    cv2.rectangle(image, (top, left), (bottom, right), (0,255,0), 2)
                
    roi = image[left:right, top:bottom]
    roi = cv2.resize(roi, (300, 300))                                        #for zooming
    cv2.imshow('ROI', roi)

    roi = cv2.resize(roi, (169, 169))                                       # Resizing back
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi = cv2.GaussianBlur(roi, (1, 1), 0)
    
    if num_frames < 10:
        if bckimg is None:
            bckimg = roi   
        cv2.accumulateWeighted(roi, bckimg.astype("float"), 0.8)             #weighted average of pics only takes float values
    
    diff = cv2.absdiff(bckimg.astype("uint8"), roi)                      #takes difference with those averaged pics
    diff = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)[1]            #thresholding GRAY img
    diff = cv2.morphologyEx(diff, cv2.MORPH_OPEN, kernalOpen)            #MORPHING for ignoring small dots
    diff = cv2.morphologyEx(diff, cv2.MORPH_CLOSE, kernalClose)          #filling blank space in hand
    diff = cv2.resize(diff, (350, 350))
    return diff
    
if __name__ == "__main__":
       
    while True:
        ret, image = camera.read()
        diffImg = returnDifference()
        
        num_frames += 1    
                
        cv2.imshow('differenceImage', diffImg)
        diffImg = cv2.resize(diffImg, (128, 128)) 
            
        result = loaded_model.predict(diffImg.reshape(1, 128, 128, 1))  
        
        prediction = {'HI'       :result[0][0], 
                      'THUMBS UP':result[0][1], 
                      'VICTORY'  :result[0][2],
                      'NICE'     :result[0][3],
                      'MY TURN'  :result[0][4],
                      'HIT'      :result[0][5]}  
                      
        # Sorting based on top prediction
        prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
        '''
        # Displaying the predictions
        cv2.putText(image, 'HI             ' +str(result[0][0]*100), (10, 120), font, 1, color[1], 1)
        cv2.putText(image, 'THUMBS UP   '    +str(result[0][1]*100), (10, 140), font, 1, color[1], 1)
        cv2.putText(image, 'VICTORY      '   +str(result[0][2]*100), (10, 160), font, 1, color[1], 1)
        cv2.putText(image, 'NICE          '  +str(result[0][3]*100), (10, 180), font, 1, color[1], 1)
        cv2.putText(image, 'MY TURN     '    +str(result[0][4]*100), (10, 200), font, 1, color[1], 1)
        cv2.putText(image, 'HIT           '  +str(result[0][5]*100), (10, 220), font, 1, color[1], 1)
        '''
        blank_screen = np.zeros((100, 220, 3), np.uint8)
        contours = cv2.findContours(diffImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
        if num_frames > 10:
            if len(contours) == 0:
                pass
            else:
                cv2.putText(blank_screen, str(prediction[0][0]), (10, 60), font, 2, color[1], 3)
        else:
            cv2.putText(blank_screen, "WAIT", (10, 60), font, 2, color[1], 3)                   #WAIT for 10 frames
            
        cv2.imshow('Prediction', blank_screen) 
        #cv2.imshow('original Image', image)                                  #Uncomment if you want to see full image
        
        keyPressed = cv2.waitKey(1) & 0xFF                                    #ESC key to terminate and 0xFF for 64-bit computers
        
        if keyPressed == 27:
            break
        
    camera.release()
    cv2.destroyAllWindows()  