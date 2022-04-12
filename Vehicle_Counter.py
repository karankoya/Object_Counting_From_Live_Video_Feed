import cv2
import numpy as np

#Capture video
vid = cv2.VideoCapture("video.mp4")

count_line_position = 550
offset = 6 
counter = 0
minWidthRect = 80
minHeightRect = 80
#Initialze background substractor (Substracts background and shows vehicles)
bgSub = cv2.bgsegm.createBackgroundSubtractorMOG()

def handle_centers(x,y,weight,height):
    x1 = int(weight/2)
    y1 = int(height/2)
    cx = x+x1
    cy = y+y1
    
    return cx, cy

detection = []

while True:
    ret,frame = vid.read() #Read image
    grayFrame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) #Grayscale the image
    blur = cv2.GaussianBlur(grayFrame,(3,3),5) #Blur the frame
    #Apply preprocessing
    img_sub = bgSub.apply(blur)
    dilates = cv2.dilate(img_sub,np.ones((5,5)))
    kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    dilatesAda = cv2.morphologyEx(dilates,cv2.MORPH_CLOSE,kernal)
    dilatesAda = cv2.morphologyEx(dilatesAda,cv2.MORPH_CLOSE,kernal)
    contour,height = cv2.findContours(dilatesAda,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    cv2.line(frame,(10,count_line_position),(1600,count_line_position),(255,0,0),3)
    #Draw bounding boxes
    for (i,c) in enumerate(contour):
        (x,y,width,height) = cv2.boundingRect(c)
        val_contour = (width >= minWidthRect) and (height >= minHeightRect)

        if not val_contour:
            continue

        cv2.rectangle(frame,(x,y),(x+width,y+height),(0,0,255),2)
        cv2.putText(frame,"Vehicle : "+str(counter),(x,y-20),cv2.FONT_HERSHEY_TRIPLEX,1,(255,244,0),2)

        center = handle_centers(x,y,width,height)
        detection.append(center)
        cv2.circle(frame,center,4,(0,255,0),-1)

        for (x,y) in detection:
            if y < (count_line_position+offset) and y > (count_line_position-offset):
                counter += 1
            
            cv2.line(frame,(10,count_line_position),(1600,count_line_position),(0,125,255),3)
            detection.remove((x,y))
            print("Vehicles:- ",counter)

    cv2.putText(frame,"VEHICLES : "+str(counter),(450,70),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,255),5)
    cv2.imshow("Vehicle Counter",frame)

    if not ret:
        break
    
    if cv2.waitKey(1) == 13:
        break
    
cv2.destroyAllWindows()
vid.release()