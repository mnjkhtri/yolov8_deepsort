import cv2
import numpy as np
import argparse
import torch
from ultralytics import YOLO

from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort

def parse_arguments():
    parser = argparse.ArgumentParser(description="YOLOv8")
    parser.add_argument(
        '--webcam-resolution',
        default=[1280, 720],
        nargs = 2,
        type = int,
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution

    # Set up the camera stream:
    cap = cv2.VideoCapture(0) #which device to use?
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    # Mouse call back for coordinate identification:
    def find_coordinates(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            coor = [x, y]
    cv2.namedWindow('LOCUS 2024')
    cv2.setMouseCallback('LOCUS 2024', find_coordinates) 

    # YOLO model:
    model = YOLO("yolov8s.pt")

    # DeepSort Tracker:
    cfg_deep = get_config()
    cfg_deep.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")
    deepsort = DeepSort(
        cfg_deep.DEEPSORT.REID_CKPT,
        max_dist=cfg_deep.DEEPSORT.MAX_DIST, min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
        nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP,
        max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
        max_age=cfg_deep.DEEPSORT.MAX_AGE, n_init=cfg_deep.DEEPSORT.N_INIT,
        nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
        use_cuda=True
    )

    # Counter zones:
    counter = 0
    counter_coor = (10, 30)
    exitt_s, exitt_e = 200, 300
    entry_s, entry_e = 300, 400

    # Track objects that enter the entry zone and not yet out of exit zone:
    arrival = {}

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)

        #Run the models and get useful stuffs:
        result = model(frame)[0]
        classes = result.boxes.cls.cpu().numpy().astype(int)
        bboxes = result.boxes.xyxy.cpu().numpy().astype(int)
        confidences = result.boxes.conf.cpu().numpy()

        # Draw and entry and exist zones for visualization:
        overlay = frame.copy() #Is this inefficient?
        cv2.line(overlay, (exitt_s, 0), (exitt_s, frame.shape[0]), (255, 0, 0), 1)
        cv2.line(overlay, (exitt_e, 0), (exitt_e, frame.shape[0]), (255, 0, 0), 1)
        cv2.line(overlay, (entry_s, 0), (entry_s, frame.shape[0]), (0, 255, 0), 1)
        cv2.line(overlay, (entry_e, 0), (entry_e, frame.shape[0]), (0, 255, 0), 1)

        alpha = 0.1
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        classes_t = []
        bboxes_t = []
        confidences_t = []

        # Update the tracker:
        for cls, box, conf in zip(classes, bboxes, confidences):
            # We only detect person, id of person = 0
            if conf < 0.5 or cls != 0: continue;

            #The bounding box coordinates
            x1, y1, x2, y2 = [int(i) for i in box]
            cx, cy = (x1+x2)//2, (y1+y2)//2
            bbox_w, bbox_h = abs(x1-x2), abs(y1-y2)

            classes_t.append(int(cls))
            bboxes_t.append([cx, cy, bbox_w, bbox_h])
            confidences_t.append(conf)

        # Tensorify
        classes_t = torch.tensor(classes_t)
        bboxes_t = torch.tensor(bboxes_t)
        confidences_t = torch.tensor(confidences_t)
        
        if classes_t.shape[0] == 0:
            continue
        #Ask the tracker for the unique objects in the frame and make the scene:
        outputs = deepsort.update(bboxes_t, confidences_t, classes_t, frame)
        for track in outputs:
            x1, y1, x2, y2 = [int(i) for i in track[:4]]
            id = track[-1]
            # Draw the bboxes:
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0, 0, 255), 1)

            label = f'track_id:{id}'
            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - 10 - text_size[1]), (x1 + text_size[0], y1 - 10), (0, 255, 255), -1)
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)

            #Increment the counter if the bounding box toches the line:
            if x1 >= entry_s and x1 <= entry_e: #The person is in entry zone:
                arrival[id] = (x1, y1)
            if id in arrival:
                if x1 >= exitt_s and x1 <= exitt_e:
                    counter += 1
                    del arrival[id]

        #Show the counter regardless of scene in the video:     
        cv2.putText(frame, f'Counter: {counter}', counter_coor, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)

        #Show the frame:
        cv2.imshow("LOCUS 2024", frame)

        if (cv2.waitKey(30) == 27): break
        
if __name__ == "__main__":
    main()
