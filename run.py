import math
import cv2
import cvzone
from numpy import vstack, array, empty
from sort import *
from ultralytics import YOLO

cap = cv2.VideoCapture(0)

model = YOLO("yolov8m.pt")

classNamen = [
    'bicycle', 'car', 'motorcycle','bus','truck',
]

vehicle_Klasse =[1, 2, 3, 5, 7]

tracker = Sort(max_age=400,min_hits=3,iou_threshold=0.3)

limits = [430,600,1550,600]

DPI = 300
def pixels_to_cm2(area_in_pixels, dpi):
    pixels_per_cm = dpi / 2.54
    cm2_per_pixel = (1 / pixels_per_cm) ** 2
    area_in_cm2 = area_in_pixels * cm2_per_pixel
    return area_in_cm2


while True:
    current_Klassen = []
    success,img = cap.read()

    img = cv2.resize(img,(1600,900))
    result = model(img,stream=True)

    detections = empty((0, 5))

    for r in result:

        boxes = r.boxes
        for box in boxes:

            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)

            w,h = x2-x1,y2-y1
            bbox = x1,y1,w,h

            conf = math.ceil((box.conf[0]*100))/100
            cls = int(box.cls[0])


            if cls in vehicle_Klasse:

                currentClass = classNamen[vehicle_Klasse.index(cls)]
                cvzone.cornerRect(img,bbox,l=9,rt=5)
                currentArray = array([x1, y1, x2, y2, conf])
                detections = vstack((detections, currentArray))
                current_Klassen.append(currentClass)

                cx, cy = x1 + w // 2, y1 + h // 2

                resultsTracker = tracker.update(detections)
                print("First")
                print()
                print()
                print()

                for result in resultsTracker:

                    print("Nested")
                    x1, y1, x2, y2, id = result
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    w, h = x2 - x1, y2 - y1
                    area = w * h
                    area_in_cm2 = pixels_to_cm2(area, DPI)
                    realeWert_cm2 = 12385.125 / area_in_cm2
                    cvzone.putTextRect(img,f'{realeWert_cm2}',(max(0,x1),max(35,y1)),scale=0.8,thickness=1,offset=3)


    cv2.imshow("Image",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cap.destroyAllWindows()