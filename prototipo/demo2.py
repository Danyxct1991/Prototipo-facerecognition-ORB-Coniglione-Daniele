from typing import Counter
import argparse
import face_recognition
import os, sys
import cv2
import numpy as np
import math
import imutils
from matplotlib import pyplot as plt
from imutils.object_detection import non_max_suppression
import pytesseract
from collections import namedtuple

from pytesseract import Output

mkey = 0


def houghSpace(im):

    img_cpy = im.copy()

    #canny edge detector
    im = cv2.Canny(im, 100, 150)

    #applicazione del calcolo dello spazio di hough all'immagine di input
    maxTheta = 180
    #la larghezza dello spazio corrisponde alla massima angolatura presa in considerazione
    houghMatrixCols = maxTheta
    
    #dimensioni dell'immagine originale
    h, w = im.shape
    #non puo' esistere nell'immagine una distanza superiore alla diagonale
    rhoMax = math.sqrt(w * w + h * h) 
    #l'altezza dello spazio è il doppio della rho massima, per considerare anche 
    #le rho negative
    houghMatrixRows = int(rhoMax) * 2 + 1 # +1 perchè l'indice parte da 0
    #le rho calcolate verranno traslate della metà dell'altezza per poter rappresentare nello spazio
    #anche le rho negative
    rhoOffset = houghMatrixRows/2
    
    #riscalature per passare da angoli a radianti
    degToRadScale = 0.01745329251994329576923690768489 # Pi / 180
    #$seno e coseno precalcolati
    rangemaxTheta = range(0,maxTheta)
    sin, cos = zip(*((math.sin(i * degToRadScale), math.cos(i * degToRadScale)) for i in rangemaxTheta))
    
    #inizializzazione dello spazio
    houghSpace = [0.0 for x in range(houghMatrixRows * houghMatrixCols)]
    
    #scorro tutta l'immagine originale
    for y in range(0, h):
        for x in range(0, w):
            #per ogni punto di bordo
            if im[y, x] > 0:
                #calcolo il suo fascio di rette...
                for theta in rangemaxTheta:
                    #... per ogni angolazione theta nello spazio, calcolo il relativo valore di rho
                    #... utilizzando la forma polare dell'equazione della retta
                    rho = int(round(x * cos[theta] + y * sin[theta] + rhoOffset))
                    
                    #una volta note le coordinate theta e rho, incremento il contatore dello spazio di Hough
                    # alla coordinata
                    c = rho * houghMatrixCols + theta
                    houghSpace[c] = houghSpace[c] + 1
    
    # normalizzazione tra 0 e 1
    m = np.max(houghSpace)
    houghSpace = houghSpace / m
    hSpace = np.reshape(houghSpace , (houghMatrixRows, houghMatrixCols))

    #filtraggio dei picchi
    hSpace[hSpace < 0.8] = 0
    #calcolo dell'istogramma
    hist = sum(hSpace)
    #calcolo dell'angolo perpendicolare
    theta1 = 90 - np.argmax(hist)
    

    #rotazione dell'immagine
    h, w, d = img_cpy.shape
    rotation_M = cv2.getRotationMatrix2D((w / 2, h / 2), -theta1, 1)
    rotated_im = cv2.warpAffine(img_cpy, rotation_M, (w,h), flags=cv2.INTER_CUBIC)

    #scrittura su disco
    #cv2.imwrite(r'rotated.jpg', rotated_im)
    return rotated_im

def funzOCR(imgali):
   

   # Apply OCR on the cropped image
    imgali = cv2.cvtColor(imgali,cv2.COLOR_BGR2GRAY)
    im_templ = cv2.imread('templpate.jpg',cv2.COLOR_BGR2GRAY)
    # uso un template per ridimensionare l'immagine e prendere le coordinate dei valori da estrarre
    des_height = im_templ.shape[0] 
    r = des_height / float(imgali.shape[0])
    dim = (int(imgali.shape[1]*r), des_height)
    imgali = cv2.resize(imgali, dim, interpolation = cv2.INTER_AREA)
    OCRLocation = namedtuple("OCRLocation", ["id", "bbox","filter_keywords"])
    # define the locations of each area of the document we wish to OCR
    OCR_LOCATIONS = [
	OCRLocation("first_name", (160, 40, 100, 16),["1."]),
	OCRLocation("last_name", (160, 64, 105, 17),["2."]),
        ]
    cconfig = ('-l ita --psm 6')
    img = imgali.copy()
    img = cv2.putText(img,"Analisi in corso", (15,15), cv2.FONT_HERSHEY_DUPLEX, 0.8, (25, 35, 55), 2)
    cv2.imshow("Analisi",img)
    for loc in OCR_LOCATIONS:
	    # extract the OCR ROI from the aligned image
        (x, y, w, h) = loc.bbox
        roi = imgali[y:y + h, x:x + w]
        # OCR the ROI using Tesseract
        text = pytesseract.image_to_string(roi,config=cconfig)
        print(text)
       
  



def face_confidence(face_distance, face_match_threshold=0.6):
    range = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + '%'
    else:
        value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2)) + '%'


def approximate_contour(contour):
    peri = cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, 0.1 * peri, True)



def get_receipt_contour(receipt_contour):    
    # loop over the contours
    for c in receipt_contour:
        approx = approximate_contour(c)
        # if our approximated contour has four points, we can assume it is receipt's rectangle
        if len(approx) == 4:
            x,y,z,b = cv2.boundingRect(receipt_contour)
            img = cv2.drawContours(img, [c], -1, (0,255,0), 3)
            return x,y,z,b

def shape_detector(c):
	shape = "unidentified"
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.04 * peri, True)
	# if the shape is a triangle, it will have 3 vertices
	if len(approx) == 3:
		shape = "triangle"
	# if the shape has 4 vertices, it is either a square or
	# a rectangle
	elif len(approx) == 4:
		# compute the bounding box of the contour and use the
		# bounding box to compute the aspect ratio
		(x, y, w, h) = cv2.boundingRect(approx)
		ar = w / float(h)
		# a square will have an aspect ratio that is approximately
		# equal to one, otherwise, the shape is a rectangle
		shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"
	print('SHAPE:\t',shape)

def funzcard(imgcid):
                kernel = np.ones((2,2),np.uint8)
                width2 = int(imgcid.shape[1])
                height2 = int(imgcid.shape[0])
                img_cpy = imgcid.copy()
                framegray = cv2.cvtColor(imgcid, cv2.COLOR_BGR2GRAY)
                #cv2.equalizeHist(framegray)
                frameblur = cv2.GaussianBlur(framegray,(5,5),0)
                frameblur= cv2.medianBlur(frameblur, 3) 
                _,threshold = cv2.threshold(frameblur,176,255,0)
            #to get outer boundery only     
                thresh = cv2.morphologyEx(threshold, cv2.MORPH_GRADIENT, kernel)
          #      cv2.imshow("threshold",threshold)
                #to strength week pixels
                thresh = cv2.dilate(thresh,kernel,iterations = 5)
                #cv2.imshow("thresh",thresh)
                contours, _ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                cv2.putText(imgcid, "Inserire Documento nello schermo", (100,50), cv2.FONT_HERSHEY_DUPLEX, 0.8, (25, 35, 255), 4)
                if len(contours)>0:
        # find the biggest countour (c) by the area
                    c = max(contours, key = cv2.contourArea)
                   # shape_detector(c)
                    x,y,w,h = cv2.boundingRect(c)
                    #mapped_c =
                   # print(len(c))
                    #cv2.rectangle(imgcid,(x,y),(x+w,y+h),(255,0,0),2)
                    peri = cv2.arcLength(c, True)
                    #approx = cv2.approxPolyDP(c, 0.04 * peri, True)
                    approx = cv2.approxPolyDP(c, 0.07 * peri, True)
                    c_area = cv2.contourArea(c)
                    cv2.putText(imgcid, str(c_area), (60,height2+50), cv2.FONT_HERSHEY_DUPLEX, 0.8, (25, 35, 255), 2)
                    cv2.imshow("Richiesta Documento",imgcid)
                    if len(approx) == 4 and c_area>=18000:
                        # print('Rectangular shape!')
                        cv2.rectangle(imgcid,(x,y),(x+w,y+h),(0,200,0),2)
                        #cv2.drawContours(imgcid, [approx], -1, (0, 200, 0), 3)
                        cv2.imshow("Richiesta Documento",imgcid)
                        imgali = img_cpy[y:y+h,x:x+w]
                        #raddrizzamento immagine
                        imgali = houghSpace(imgali)
                        funzOCR(imgali)
                 #   cv2.drawContours(imgcid, [c], -1, (0, 255, 0), 3)
               #    img=cv2.imwrite('carta.jpg',imgcid)
                        


class FaceRecognition:
    face_locations = []
    face_encodings = []
    face_names = []
    known_face_encodings = []
    known_face_names = []
    process_current_frame = True

    def __init__(self):
        self.encode_faces()

    def encode_faces(self):
        for image in os.listdir('faces'):
            face_image = face_recognition.load_image_file(f"faces/{image}")
            face_encoding = face_recognition.face_encodings(face_image)[0]

            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(image)
        print(self.known_face_names)


    def run_recognition(self):
        video_capture = cv2.VideoCapture(0)
        # ho utilizzato ip webcam app android per avere una cam con risoluzione migliore
        #video_capture = cv2.VideoCapture('http://192.168.139.253:8090/video')
        #video_capture = cv2.VideoCapture('mariotc.mp4')
        

        if not video_capture.isOpened():
            sys.exit('Video source not found...')

        while True:
            ret, frame = video_capture.read()

            if ret == False:
                break
            
                    
            frame = imutils.resize(frame, width=750)
            imgcid = frame.copy()
            # key access funzcard()
            key=2
            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]

                # Find all the faces and face encodings in the current frame of video
            self.face_locations = face_recognition.face_locations(rgb_small_frame)
            self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)
            self.face_names = []
            for face_encoding in self.face_encodings:
                        # See if the face is a match for the known face(s)
                        matches1 = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                        name = "Unknown"
                        confidence = '???'
                        key = 0     

                        # Calculate the shortest distance to face
                        face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                       

                        best_match_index = np.argmin(face_distances)
                        if matches1[best_match_index]:
                                name = self.known_face_names[best_match_index]
                                confidence = face_confidence(face_distances[best_match_index])
                                if self.known_face_names[best_match_index] != "Unknown":
                                    key=1

                        self.face_names.append(f'{name} ({confidence})')
                         


            # Display the results
            for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                # Create the frame with the name
                
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                idface=cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
                    
                
                
                
            # Display the resulting image
            cv2.imshow('Face Recognition', frame)
            global mkey
            if(key == 1):
                img=cv2.imwrite('riconosciuto.jpg',frame)
            if(key == 0):
                if(mkey == 0):
                    mkey = 1
                    #cv2.imwrite('non riconosciuto.jpg',frame)
                    cv2.imshow("non riconosciuto",frame)
                    video_capture.release()
                    print("Viso Sconosciuto rilevato, inizio ricerca di un documento d'identità")
            if(mkey == 1):
                video_capture = cv2.VideoCapture('http://192.168.139.253:8090/video')

                funzcard(imgcid)
            

            key = cv2.waitKey(1)
            # Hit 'q' on the keyboard to quit!
            if key == ord('q'):
                break
            #pause
            elif key ==ord('p'):
                cv2.waitKey()
            #reload
            elif key ==ord('r'):
                mkey = 0
                cv2.destroyWindow("non riconosciuto")
                cv2.destroyWindow("Richiesta Documento")
           
                
                

        # Release handle to the webcam
        video_capture.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
   fr = FaceRecognition()
   fr.run_recognition()
 
