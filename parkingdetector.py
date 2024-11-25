import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import time

def detect_parking_availability(video_path='parking1slow.mp4', model_path='yolov8s.pt'):
    # Load the model
    model = YOLO(model_path)
    
    # Load video
    cap = cv2.VideoCapture(video_path)
    
    # Load class list
    with open("coco.txt", "r") as f:
        class_list = f.read().splitlines()
    
    # Define parking areas
    areas = [
        [(52, 364), (30, 417), (73, 412), (88, 369)],
        [(105, 353), (86, 428), (137, 427), (146, 358)],
        [(159,354),(150,427),(204,425),(203,353)],
        [(217,352),(219,422),(273,418),(261,347)],
        [(274,345),(286,417),(338,415),(321,345)],
        [(336,343),(357,410),(409,408),(382,340)],
        [(396,338),(426,404),(479,399),(439,334)],
        [(458,333),(494,397),(543,390),(495,330)],
        [(511,327),(557,388),(603,383),(549,324)],
        [(564,323),(615,381),(654,372),(596,315)],
        [(616,316),(666,369),(703,363),(642,312)],
        [(674,311),(730,360),(764,355),(707,308)],
    ]

    # List to store results
    availability = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.resize(frame, (1020, 500))
        results = model.predict(frame)
        detections = results[0].boxes.data
        detection_df = pd.DataFrame(detections).astype("float")
        
        area_occupancy = [0] * len(areas)  # Initialize all areas as unoccupied
        #occupied_spaces = []

        for _, row in detection_df.iterrows():
            x1, y1, x2, y2, _, class_id = map(int, row)
            class_name = class_list[class_id]
            
            if 'car' in class_name:
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                for i, area in enumerate(areas):
                    if cv2.pointPolygonTest(np.array(area, np.int32), (cx, cy), False) >= 0:
                        area_occupancy[i] += 1
                       # occupied_spaces.append(i+1)
        
       # availability.append({
        #    "total_spaces": len(areas),
         #   "free_spaces": len(areas) - sum(1 for a in area_occupancy if a > 0),
           # "occupied_space_indices": occupied_spaces,
          #  "latitude": 32.748615611210546,
           # "longitude": -97.08889801033872,
        #})
        yield {
             "total_spaces": len(areas),
            "free_spaces": len(areas) - sum(1 for a in area_occupancy if a > 0),
            "latitude": 32.748615611210546,
            "longitude": -97.08889801033872,
        }
  
  
    cap.release()
    
