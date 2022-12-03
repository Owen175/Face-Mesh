import mediapipe as mp
import cv2
import time

#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('Videos/2.mp4')
pTime = 0
print(cap.read())
mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)

    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_CONTOURS, drawSpec, drawSpec)
            # ih, iw, ic = img.shape
            # lm = faceLms.landmark[index]
            # x, y = int(lm/x*iw), int(lm.y*ih)
            # Indices for the different face points
            # https://github.com/ManuelTS/augmentedFaceMeshIndices

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 3)
    cv2.imshow('Image', img)
    cv2.waitKey(1)