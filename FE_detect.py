import cv2
import sys

# cascPath = sys.argv[1]
face = cv2.CascadeClassifier('F:\openCV\cascades\haarcascade_frontalface_default.xml')
eye=cv2.CascadeClassifier('F:\openCV\cascades\haarcascade_eye.xml')
vid = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = vid.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=10,
        minSize=(20, 20),
        # flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    #Now detect eyes
        roi_color = frame[y:y+h, x:x+w]
        roi_gray = gray[y:y+h, x:x+w]

        # roi_gray=cv2.cvtColor(roi_color,cv2.COLOR_BGR2GRAY)

        eyes = eye.detectMultiScale(roi_gray,1.2,3)
    
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,0,0),2)
            # cv2.circle(roi_color,(ex,ey),ew,(255,0,0),2)
            
        

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(2) == ord('q'):
        break

# When everything is done, release the capture
vid.release()
cv2.destroyAllWindows()