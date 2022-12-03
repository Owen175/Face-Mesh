import mediapipe as mp
import cv2
import time


class FaceMeshDetector():
    def __init__(self,
               staticMode=False,
               maxFaces=2,
               refineLandmarks=False,
               minDetectionCon=0.5,
               minTrackCon=0.5):
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon
        self.refineLandmarks = refineLandmarks

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(
               static_image_mode=self.staticMode,
               max_num_faces=self.maxFaces,
               refine_landmarks=self.refineLandmarks,
               min_detection_confidence=self.minDetectionCon,
               min_tracking_confidence=self.minTrackCon)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)

    def findFaceMesh(self, img, draw=True, indexes=[], returnWholeFace=False, numberedPoints=False):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(imgRGB)
        self.indexList = []
        self.face = []
        self.faces = []
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if returnWholeFace or numberedPoints:
                    for id, lm in enumerate(faceLms.landmark):
                        ih, iw, ic = img.shape
                        x, y = int(lm.x * iw), int(lm.y * ih)
                        if returnWholeFace:
                            self.face.append([x, y])
                        if numberedPoints:
                            cv2.putText(img, f'{str(id)}', (x, y), cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 255, 0), 1)
                for index in indexes:
                    ih, iw, ic = img.shape
                    lm = faceLms.landmark[index]
                    x, y = int(lm.x*iw), int(lm.y*ih)
                    self.indexList.append([x,y])
                    # Indices for the different face points
                    # https://github.com/ManuelTS/augmentedFaceMeshIndices
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS, self.drawSpec, self.drawSpec)

            self.faces.append(self.face)
        if returnWholeFace is False:
            return img, self.indexList
        else:
            return img, self.faces




def main():
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture('Videos/4.mp4')
    detector = FaceMeshDetector()
    pTime = 0

    while True:
        success, img = cap.read()
        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime

        #img, indexList = detector.findFaceMesh(img, indexes=[23,34,5,4,6,344,34,56,43])
        img, faceList = detector.findFaceMesh(img, returnWholeFace=True, numberedPoints=True)

        cv2.putText(img, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 3)
        cv2.imshow('Image', img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()