def refind():
    
    #global bboxm
    global okm
    global trackerm
    
    ret,frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = np.array(face_cascade.detectMultiScale(gray, 1.3, 5))
    #cv2.imshow('old_frame1',frame)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        cap.release()
        cv2.destroyAllWindows()
        sys.exit()


    facem=[]
    if len(faces)!=0:
        faces_x=faces[:,0]
        faces_y=faces[:,1]
        faces_w=faces[:,2]
        faces_h=faces[:,3]
        #print "faces_x", faces_x
        #print "faces_w", faces_w
        #print "faces_x+faces_w", faces_x+faces_w
        
        #face1 is the image of extracted face
        for x,y,w,h in zip(faces_x,faces_y,faces_w,faces_h):
            face1=frame[y:y+h, x:x+w]
            facem.append(face1)
            cv2.imshow('face',face1)
        #print "faces", faces
        #print "face1:", face1
        print "facem len:", len(facem) 
        
        trackerm=[]
        okm=[]
        #bboxm=[]
        for x,y,w,h in zip(faces_x,faces_y,faces_w,faces_h):
            tracker = cv2.TrackerKCF_create()
            bbox =  (x,y, w, h) 
            ok = tracker.init(frame, bbox)
            trackerm.append(tracker)
            okm.append(ok)
        #okm=np.array(okm)

    else:
        print "no faces found in refind"
    

#haarCascade Face and Eye detector
import numpy as np
import cv2
import sys
import time

face_cascade = cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_eye.xml')
cap = cv2.VideoCapture(0)
cap.release()
cap=cv2.VideoCapture(0)
cv2.destroyAllWindows()

#initial detection of faces
while(1):
    ret,frame = cap.read()
    frame = np.array(frame)
    print type(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = np.array(face_cascade.detectMultiScale(gray, 1.3, 5))
    cv2.imshow('output',frame)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        cap.release()
        cv2.destroyAllWindows()
        sys.exit()


    #x y w h for 1 face and face1 image
    #facem stores the multiple face frames
    facem=[]
    if len(faces)!=0:
        faces_x=faces[:,0]
        faces_y=faces[:,1]
        faces_w=faces[:,2]
        faces_h=faces[:,3]
        print "faces_x", faces_x
        print "faces_w", faces_w
        print "faces_x+faces_w", faces_x+faces_w
        #face1 is the image of extracted face
        for x,y,w,h in zip(faces_x,faces_y,faces_w,faces_h):
            face1=frame[y:y+h, x:x+w]
            facem.append(face1)
            cv2.imshow('face',face1)
        #print "faces", faces
        #print "face1:", face1
        print "facem len:", len(facem) 
        #if len(facem)>1:
        cv2.imshow('face1', facem[0])
        print "face1:dtype" , type(facem[0])
        print "face1", facem[0]
        break










#kcf tracker initialisation
trackerm=[]
okm=[]
for x,y,w,h in zip(faces_x,faces_y,faces_w,faces_h):
    tracker = cv2.TrackerKCF_create()
    bbox =  (x,y, w, h) 
    ok = tracker.init(frame, bbox)
    trackerm.append(tracker)
    okm.append(ok)

#print "ok size:", len(ok)
#sys.exit()

fps=0
fps_counter=0
timer=time.time()
frames=0

while True:
    # Read a new frame
    ok, frame = cap.read()
    if not ok:
        break

    # Update tracker
    #ok, bbox = tracker.update(frame)
    okm=[]
    bboxm=[]
    for tracker in trackerm:
        ok, bbox = tracker.update(frame)
        okm.append(ok)
        bboxm.append(bbox)

    # Draw bounding box
    okm=np.array(okm)
    bboxm=np.array(bboxm)
    for box_x, box_y, box_w, box_h in zip(bboxm[okm==True][:,0], bboxm[okm==True][:,1],bboxm[okm==True][:,2],bboxm[okm==True][:,3]): #try boxm[ok==True][0] for box_x .... 
        p1 = (int(box_x), int(box_y))
        p2 = (int(box_x + box_w), int(box_y + box_h))
        cv2.rectangle(frame, p1, p2, (0,255,0), 2, 1)
    
   # Exit if ESC pressed
    k = cv2.waitKey(1) & 0xff
    if k == 27 : 
        cv2.destroyAllWindows()
        cap.release()
        break
        
    
    #timer,fps, refind:finds the face and features again
    fps_counter=fps_counter+1
    if(time.time()-timer>1):
        print timer,":",fps_counter
        fps=fps_counter
        fps_counter=0
        timer=time.time()
        #removing is ok cond gives good fit bounding box but reduces fps , significant if box is large 
        #if ok==0:
        print "before refine bbox:", bboxm
        refind()
        print "after refine bbox:", bboxm
        #tracker = cv2.TrackerKCF_create()
        #ok = tracker.init(frame, bbox)
        #print "refined ok", ok

    

    #frames, refind:finds the face and features again
    frames=frames+1
    if(frames>fps):
        frames=0
        #refind()
    cv2.putText(frame, "Fps:"+str(fps), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(255,0,0),2)
                
    #output
    cv2.imshow('output', frame)


