import cv2
from mtcnn.mtcnn import MTCNN
from deepface import DeepFace
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

class FaceTracker:
    def __init__(self):
        self.detector = MTCNN()
        self.tracker = DeepSort()

    def track_faces(self):
        video_capture = cv2.VideoCapture("videoplayback.mp4") 

        while True:
            ret, frame = video_capture.read()

            if not ret:
                break

            face_boxes, face_features, face_score = self.detect_faces(frame)
            if face_boxes:
                tracks = self.track_detected_faces(face_boxes, face_features, face_score, frame)
                self.display_tracking_info(tracks, frame)
            cv2.imshow('Video', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        video_capture.release()
        cv2.destroyAllWindows()

    def detect_faces(self, frame):
        faces = self.detector.detect_faces(frame) # this function will use MTCNN to detect all the faces and analyse the face emotions
        face_features = []
        face_boxes = []
        face_score = []

        for face in faces:
            x, y, width, height = face['box']
            print("first x, y, width, height :" ,x, y, width, height )
            face_image = frame[y:y+height, x:x+width]
            #cv2.imshow('Video', face_image)
            demographies = DeepFace.analyze(face_image, enforce_detection=False) # Analyze the face emotions using deepface
            emotions = demographies[0]['emotion']
            max_emotion, max_value = max(emotions.items(), key=lambda x: x[1])

            face_boxes.append(([x, y, width, height], max_value, max_emotion))
            face_features.append(face['keypoints'])

        return face_boxes, face_features, face_score
    # track the faces using deep sort by given face id ,and expression
    def track_detected_faces(self, faces, face_features, face_score, frame):
        if faces:
            face_features = np.array(face_features) 
            face_score = np.array(face_score)
            tracks = self.tracker.update_tracks(faces, frame=frame)
            return tracks
    #desplay the tracked info including (Id,Face Expression ,Score)
    def display_tracking_info(self, tracks, frame):
        for track in tracks:
            if not track.is_confirmed():
                continue

            bbox = list(track.to_tlbr()) 
            txt = 'id:' + str(track.track_id)
            txt_emotion = 'emotion:' + str(track.det_class)
            txt_confi = 'condfidence:' + str(track.det_conf)
            print("second x, y, width, height :",bbox )
            (label_width,label_height), baseline = cv2.getTextSize(txt , cv2.FONT_HERSHEY_SIMPLEX,1,1)
            baseline = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)[1]
            top_left = tuple(map(int, [int(bbox[0]), int(bbox[1]) - (label_height + baseline)]))
            top_right = tuple(map(int, [int(bbox[0]) + label_width, int(bbox[1])]))
            org = tuple(map(int, [int(bbox[0]), int(bbox[1]) - baseline]))
            org1 = tuple(map(int, [int(bbox[0]), int(bbox[1] + 20) - baseline]))
            org2 = tuple(map(int, [int(bbox[0]), int(bbox[1] + 45) - baseline]))

            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), int((bbox[2]), int(bbox[3])), (255, 0, 0), 1)
            cv2.rectangle(frame, top_left, top_right, (255, 0, 0), -1)
            cv2.putText(frame, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
            cv2.putText(frame, txt_emotion, org1, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
            cv2.putText(frame, txt_confi, org2, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

if __name__ == "__main__":
    tracker = FaceTracker()
    tracker.track_faces()
