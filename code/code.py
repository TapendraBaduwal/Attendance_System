import cv2
import numpy as np
import face_recognition #install C++ related library from VS-code then #pip install cmake,pip install dlib, then, pip3 install face_recognition
import os
import csv
from datetime import datetime
path="images"  #path of images
images=[]      #creat list to store images
personName=[]  #creat list to store name of person
myList=os.listdir(path) #pass images directory to read images,listing current directory images
print(myList) #run and check the output of images name
for cu_img in myList: #read all images through cu_img variable and split name text
    current_imag=cv2.imread(f"{path}/{cu_img}") #use cv2 module to read image and give path under cu_img list and store in current_img variable through f string 
    images.append(current_imag) #images list append in images[] list
    personName.append(os.path.splitext(cu_img)[0]) #name append in personName[] through os.path and split value come from this path,0th element  are name of cu_img

print(personName)#run and check the output as name
#Define generalize function to add N number of images
def faceEncodings(images): #Face_recognition-models based on  dlib which Encodes face into 128 different features
    encodeList=[] #empty list then append encode
    for img in images:  #receives images access through img usig for loop
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB) #convert receives img to RGB
        encode=face_recognition.face_encodings(img)[0]#img encoding and first element given as 0
        encodeList.append(encode)
    return encodeList
encodeListknown=faceEncodings(images)#call function by passing images and store in variable
print(" Hello Tapendra All Encoding Complete!!!!")

#Attendance function and creat attendance.csv file similar as code.py created or download sample csv file upload in project file
def attendance(name): # attendace base upon name so pass name
    with open("/home/madan/Desktop/Attendance_System-Using_Face_Recognition/csv/attendance.csv","r+") as f: #with open read ,r+ is read append mode,object f,path of csv file or csv file name.csv
        myDataList=f.readlines() #read and contain datalist
        nameList=[] #empty list to append the data
        for line in myDataList: # for loop apply for split values with comma,
            entry=line.split(",") #split values like name, date with comma
            nameList.append(entry[0]) #append  name in empty list
        if name not in nameList: # if name is  not in namelist that means still that name entry is still not pass so now pass the entry to record data
            time_now=datetime.now() # read record of date  and time  both and save in variable
            tstr=time_now.strftime("%H:%M:%S") # now function record current date and time,record hour minute
            dstr=time_now.strftime("%D/%M/%Y") #again  record day,month,year
            f.writelines(f"\n{name},{tstr},{dstr}") # write through f string format with writelines function



# Now read camera and 0 is the camera id of laptop 
cap=cv2.VideoCapture(0)
while True: #video is the sequence of images so while infinit loop then then break
    ret,frame=cap.read() #call read through cap and access through fram and hold by ret variable
    faces=cv2.resize(frame,(0,0),None,0.25,0.25)#resize image come from frame ,destination None,0.25 is 1/4 part
    faces=cv2.cvtColor(faces,cv2.COLOR_BGR2RGB)#again convert faces come from camera to RGB

    facesCurrentFrame=face_recognition.face_locations(faces) #face location find out and store in variable
    encodesCurrentFrame=face_recognition.face_encodings(faces,facesCurrentFrame)#Face encoding find out and store in variable
    
    #use above  facesCurrentFrame and encodesCurrentFrame to fine out face matching or not and find out face distance
    for encodeFace,faceLoc in zip(encodesCurrentFrame,facesCurrentFrame): #Zip function use for pass two or more parameter in  one package and read through encodeFace,faceLoc
        matches=face_recognition.compare_faces(encodeListknown,encodeFace)#compare faces
        faceDis=face_recognition.face_distance(encodeListknown,encodeFace)#find out face distance less distance match,high distance not match
        
        matchIndex=np.argmin(faceDis)#give index value of minimum distance
        #if image come from camera present in our image  directory then pick name and give person present in our directory
        if matches[matchIndex]:
            name=personName[matchIndex].upper() #if matchIndex  value present in personName list then take in upper(capital ltr all) case and visualize
            #print(name) run and test
            y1,x2,y2,x1=faceLoc #faceLock pattern in facial_recognation library 
            y1,x2,y2,x1=y1*4,x2*4,y2*4,x1*4 #resize 1/4th already in above so now multiply by 4 to bring in orginal size
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)#apply rectangle on frame and apply green color
            cv2.rectangle(frame,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED) #To print name on small rectangle shape under face rectangle with suitable data
            cv2.putText(frame,name,(x1 +6,y2 -6),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2) #put Name under smaller rectangle with suitable data
            attendance(name) #calling function
            
       
    cv2.imshow("Camera",frame)
    if cv2.waitKey(1)==13: #13 is askey key of Enter,press enter to stop camera
        break
cap.release() 
cv2.destroyAllWindows()
print("Hello Tapendra your project complet sucessfully!!!!")

