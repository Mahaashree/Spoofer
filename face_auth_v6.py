#WELCOME TO VERSION 6
#THIS IS AN UPDATE FROM V5, 
#NOW WE GET INPUT USER'S IDENTIY AND GRANT ACCESS


import cv2
import face_recognition as fr
import numpy as np
import os


class FaceAuth:
    def __init__(self, known_faces_dir):
        self.known_encoding =[]
        self.known_names = []
        self.authorized_users = {}
        self.load_known_faces(known_faces_dir)

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

    
    def live_auth(self, target_name, tolerance = 0.45):
        #opening cam
        cap = cv2.VideoCapture(0)

        #Auth rules
        max_attempts = 10
        attempts = 0


        while attempts<max_attempts:
            #capturing frame by frame
            ret, frame = cap.read()
            if not ret:
                break

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




