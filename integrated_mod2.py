#WELCOME TO INTEGRATED_MOD
#INTEGRATING V6 AND SPOOFER - ATTEMPT 2


import cv2
import face_recognition as fr
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D
from collections import deque


class FaceAuth:
    def __init__(self, known_faces_dir, anti_spoof_model = 'antispoofing_full_model.h5'):
        self.known_encoding =[]
        self.known_names = []
        self.authorized_users = {}

        #Loading known faces
        self.load_known_faces(known_faces_dir)

        #initialize Anti-spoof
        self.face_cascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")

        def custom_depthwise_conv2d(**kwargs):
            kwargs.pop('groups', None)
            return DepthwiseConv2D(**kwargs)
        
        #Loading anti spoof model
        self.anti_spoof_model = load_model(anti_spoof_model, custom_objects = {
            'DepthwiseConv2D': custom_depthwise_conv2d})
        
        print("Models loaded successfully (haarcascade, spoofer)")

        #THRESHOLDS
        self.spoof_thresh = 0.4 #below this is considered spoof
        self.frame_history = 20 #Number of frames we're considering
        self.spoof_confidence = 0.8 # allow max 80% of spoof frames


    def load_known_faces(self, known_faces_dir):

        for filename in os.listdir(known_faces_dir):
            if filename.endswith(('.jpg','.jpeg','.png')):
                img_path = os.path.join(known_faces_dir, filename)
                

                #get face encoding
                try:
                    img = fr.load_image_file(img_path)
                    
                    face_loc = fr.face_locations(img)

                    if not face_loc:
                        print(f"No face detected in {filename}")
                        continue

                    encodings = fr.face_encodings(img, face_loc)
                    if encodings:
                        name = os.path.splitext(filename)[0]
                        self.known_encoding.append(encodings[0])
                        #get the name
                        self.known_names.append(name)

                        self.authorized_users[name] = {
                            'encoding': encodings[0],
                            'access_level':'standard'
                        }
                
                except Exception as e:
                    print(f"Error processing {filename}: {e}")


    def check_spoof (self, frame):
        #checks if frame has real or spoof 
        #returns true if real, false id spoof

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            return False

        spoof_preds = deque(maxlen = self.frame_history)
        
        for (x, y, w, h) in faces:
            face = frame[y-5:y+h+5, x-5:x+w+5]
            resized_face = cv2.resize(face, (160, 160))
            resized_face = resized_face.astype("float") / 255.0
            resized_face = np.expand_dims(resized_face, axis=0)
            
            # Predict using the anti-spoofing model
            preds = self.anti_spoof_model.predict(resized_face)[0][0]
            print(f"Full prediction: {preds}")
            #storing this pred
            spoof_preds.append(preds < self.spoof_thresh)
            
        if not spoof_preds:
            return False
        
        spoof_count = sum(spoof_preds)
        spoof_ratio = spoof_count / len(spoof_preds)

        print(f"Spoof ratio: {spoof_ratio}")

        return spoof_ratio > self.spoof_confidence
    
  
    def live_auth(self, target_name, tolerance = 0.45):
        #opening cam
        cap = cv2.VideoCapture(0)

        #Auth rules
        max_attempts = 20
        attempts = 0

        spoof_history = deque(maxlen = self.frame_history)


        while attempts<max_attempts:
            #capturing frame by frame
            ret, frame = cap.read()
            if not ret:
                break

            is_real = self.check_spoof(frame)
            spoof_history.append(is_real)
            #print(f"Spoof history: {spoof_history}")

            if len(spoof_history) == self.frame_history:
                spoof_ratio = spoof_history.count(False) / len(spoof_history)
                if spoof_ratio > (1-self.spoof_confidence):
                    print(f"Spoofing detected!: {spoof_ratio *100:.2f}% frame of spoof")
                    cap.release()
                    cv2.destroyAllWindows()
                    return False



                #conv bgr to rgb
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                #finding face locations and encodings in frame
                face_loc = fr.face_locations(rgb_frame)
                face_encoding = fr.face_encodings(rgb_frame, face_loc)

                #processing each face
                for (top, right, bottom, left), face_encoding in zip(face_loc, face_encoding):
                    name = "unknown"
                    access_granted = False

                    #checking if target user is in known encoding
                    if target_name in self.authorized_users:
                        #compare with specific user's encoding
                        matches = fr.compare_faces(
                            [self.authorized_users[target_name]['encoding']],
                            face_encoding,
                            tolerance=tolerance
                        )

                        #verifying identity
                        if matches[0]:
                            name = target_name
                            access_granted = True
                            print(f"Access granted to {target_name}")

                            #adding visuals for access granted
                            cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0),3)
                            cv2.putText(frame, f"Access Granted for {name}!",
                                        (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 
                                        0.9, (0,255,0),2)
                            
                            cap.release()
                            cv2.destroyAllWindows()
                            return True
                    
                    cv2.rectangle(frame, (left, top), (right,bottom),(0,0,255) if not access_granted else (0,255,0), 2)

                    #display name
                    cv2.putText(frame, name, (left+6, bottom-6),
                                cv2.FONT_HERSHEY_COMPLEX, 1.0, (255,255,255),1)
                    

            cv2.imshow('Face Auth', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            attempts +=1
        cap.release()
        cv2.destroyAllWindows()
        print(f"Access denied for {target_name}")
        return False



def main():
    known_face_dir = 'data/known/known'

    auth = FaceAuth(known_face_dir)

    target_name = input("Enter your name: ")
    result = auth.live_auth(target_name)

    if result:
        print("Welcome to spoofer!")

    else:
        print("Auth failed. Access denied")


if __name__=='__main__':
    main()




