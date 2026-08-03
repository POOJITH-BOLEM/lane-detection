from ultralytics import YOLO
import numpy as np
import cv2
from utils import *
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--video', type=str, default='project_video.mp4', help='path to video file')
parser.add_argument('--src', type=int, default=0, help='source of the camera')
parser.add_argument('--output_dir', type=str, default='', help='path to the output directory')
args = parser.parse_args()

print('----- info -----')
print('[i] Path to video file: ', args.video)
print('###########################################################\n')

frameWidth = 640
frameHeight = 480

font = cv2.FONT_HERSHEY_PLAIN
frame_id = 0
cameraFeed = False

if cameraFeed:
    intialTracbarVals = [24, 55, 12, 100]
else:
    intialTracbarVals = [42, 63, 14, 87]

if cameraFeed:
    cap = cv2.VideoCapture(args.src)
    cap.set(3, frameWidth)
    cap.set(4, frameHeight)
else:
    cap = cv2.VideoCapture(args.video)

initializeTrackbars(intialTracbarVals)

# Setup Video Writer
starting_time = time.time()
arrayCounter = 0
noOfArrayValues = 10
arrayCurve = np.zeros([noOfArrayValues])
model = YOLO("yolov8n.pt")
model.to("cuda")

while True:
    frame_id += 1
    success, img = cap.read()
    if not success:
        print('[i] ==> Done processing!!!')
        break

    img = cv2.resize(img, (frameWidth, frameHeight))
    imgFinal = img.copy()
    imgUndis = undistort(img)
    
    # 1. Lane Detection Pipeline
    imgThres, imgCanny, imgColor = thresholding(imgUndis)
    src_points = valTrackbars()
    imgWarp = perspective_warp(imgThres, dst_size=(frameWidth, frameHeight), src=src_points)
    imgSliding, curves, lanes, ploty = sliding_window(imgWarp, draw_windows=True)

    try:
        curverad = get_curve(imgFinal, curves[0], curves[1])
        lane_curve = np.mean([curverad[0], curverad[1]])
        imgFinal = draw_lanes(img, curves[0], curves[1], frameWidth, frameHeight, src=src_points)

        currentCurve = lane_curve // 50
        if int(np.sum(arrayCurve)) == 0:
            averageCurve = currentCurve
        else:
            averageCurve = np.sum(arrayCurve) // arrayCurve.shape[0]
        
        if abs(averageCurve - currentCurve) > 200:
            arrayCurve[arrayCounter] = averageCurve
        else:
            arrayCurve[arrayCounter] = currentCurve
        
        arrayCounter += 1
        if arrayCounter >= noOfArrayValues: arrayCounter = 0
        cv2.putText(imgFinal, str(int(averageCurve)), (frameWidth // 2 - 70, 70), cv2.FONT_HERSHEY_DUPLEX, 1.75, (0, 0, 255), 2)
    except:
        lane_curve = 0

    imgFinal = drawLines(imgFinal, lane_curve)

    # 2. Object Detection (YOLOv8)
    results = model(imgFinal, device=0)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            label = model.names[cls]

            if conf > 0.5:
                cv2.rectangle(imgFinal, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(imgFinal, f"{label} {conf:.2f}",
                            (x1, y1 - 10),
                            font, 1,
                            (0, 255, 0), 2)


    # 3. Visualization
    elapsed_time = time.time() - starting_time
    fps = frame_id / elapsed_time
    cv2.putText(imgFinal, f"FPS: {fps:.2f}", (10, 30), font, 2, (0, 255, 255), 2)

    imgStacked = stackImages(0.5, ([imgUndis, imgColor],
                                   [imgCanny, imgWarp],
                                   [imgSliding, imgFinal]))

    cv2.imshow("Detection Pipeline", imgStacked)
    cv2.imshow("Final Result", imgFinal)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print('==> All done!')