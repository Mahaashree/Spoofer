#WELCOME TO VERSION 5
#THE FIRST SATISFACTORILY WORKING AUTH MODEL

import cv2
import face_recognition as fr
import numpy as np

class FaceAuth:
    def __init__(self, known_faces_dir):
        self.known_encoding =[]
        self.known_names = []
        self.load_known_faces(known_faces_dir)

    def load_known_faces(self, known_faces_dir):
        import os

        for filename in os.listdir(known_faces_dir):
            if filename.endswith(('.jpg','.jpeg','.png')):
                img_path = os.path.join(known_faces_dir, filename)
                img = fr.load_image_file(img_path)

                #get face encoding
                try:
                    encodings = fr.face_encodings(img)
                    if encodings:
                        self.known_encoding.append(encodings[0])
                        #get the name
                        self.known_names.append(os.path.splitext(filename)[0])
                
                except Exception as e:
                    print(f"Error processing {filename}: {e}")

    
    def live_auth(self, tolerance = 0.45):
        #opening cam
        cap = cv2.VideoCapture(0)

        while True:
            #capturing frame by frame
            ret, frame = cap.read()
            if not ret:
                break

            #conv bgr to rgb
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            #finding face locations and encodings in frame
            face_loc = fr.face_locations(rgb_frame)
            face_encoding = fr.face_encodings(rgb_frame, face_loc)


            for (top, right, bottom, left), face_encoding in zip(face_loc, face_encoding):
                name = "unknown"

                if self.known_encoding:
                    matches = fr.compare_faces(
                        self.known_encoding,
                        face_encoding,
                        tolerance=tolerance
                    )

                face_distances = fr.face_distance(self.known_encoding, face_encoding)

                #if match is found, use first matching name
                if True in matches:
                    first_match = matches.index(True)
                    name = self.known_names[first_match]
                
                print(f"Match:{name}")
                print(f"Face distances: {face_distances}")

                
                #draw face box
                cv2.rectangle(frame, (left,top),(right, bottom),(0,0,255) if name =="unknown" else (0,255,0),2)


                #predicted name
                cv2.rectangle(frame, (left, bottom-35), (right, bottom), (0,0,255) if name =="unknown" else (0,255,0),cv2.FILLED)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, name, (left+6, bottom-6), font, 1.0, (255,255,255) if name == "unknown" else (255,0,0),1)

            cv2.imshow('Face Auth', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


def main():
    known_face_dir = 'data/known/known'

    auth = FaceAuth(known_face_dir)

    auth.live_auth()

if __name__=='__main__':
    main()




