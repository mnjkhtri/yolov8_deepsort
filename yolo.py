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
        default=[1280, 1080],
        nargs = 2,
        type = int,
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution

    def list_webcams(upto):
        # Get the list of available camera devices
        all_camera_indices = list(range(upto))  # You can adjust the range based on the number of cameras you expect

        available_cameras = []
        for index in all_camera_indices:
            # Try to open the camera with the current index
            cap = cv2.VideoCapture(index)  # Use cv2.CAP_DSHOW to avoid a potential issue on Windows

            # Check if the camera is opened successfully
            if cap.isOpened():
                available_cameras.append(index)
                cap.release()  # Release the camera capture object

        return available_cameras

    # Get the list of available webcams
    webcam_list = list_webcams(10)

    # Print the list
    print("Available webcams:", webcam_list)



    # Set up the camera stream:
    cap = cv2.VideoCapture(0) #which device to use?
    if not (cap.isOpened()):
        print("Could not open video device")
        assert(False)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    # Mouse call back for coordinate identification:
    def find_coordinates(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            coor = [x, y]
    cv2.namedWindow('LOCUS 2024')
    cv2.setMouseCallback('LOCUS 2024', find_coordinates) 

    # YOLO model:
    model = YOLO("yolov8l.pt")

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
    exitt_s, exitt_e = 550, 600
    entry_s, entry_e = 600, 650

    # Track objects that enter the entry zone and not yet out of exit zone:
    arrival = {}

    import logging, json
    logging.basicConfig(filename='./tracker.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    from datetime import datetime, timedelta
    start_time = datetime.now()
    while True:
        #LOGGING START#
        elapsed_time = datetime.now() - start_time
        if elapsed_time >= timedelta(minutes=1):
            print("Logged now at ", datetime.now())
            # Log a message with the counter value

            logging.info(json.dumps(
                {
                    "counter": counter,
                    "timestamp": str(datetime.now()),
                }
            ))

            # Reset the start time
            start_time = datetime.now()
        #LOGGING STOP#
                
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

        cv2.line(overlay, (0, exitt_s), (frame.shape[1], exitt_s), (255, 0, 0), 1)
        cv2.line(overlay, (0, exitt_e), (frame.shape[1], exitt_e), (255, 0, 0), 1)
        cv2.line(overlay, (0, entry_s), (frame.shape[1], entry_s), (0, 255, 0), 1)
        cv2.line(overlay, (0, entry_e), (frame.shape[1], entry_e), (0, 255, 0), 1)


        alpha = 0.5
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
        
        if classes_t.shape[0] != 0:
            #Ask the tracker for the unique objects in the frame and make the scene:
            outputs = deepsort.update(bboxes_t, confidences_t, classes_t, frame)
            for track in outputs:
                x1, y1, x2, y2 = [int(i) for i in track[:4]]
                print(track)
                id = track[-2]
                # Draw the bboxes:
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0, 0, 255), 1)

                label = f'track_id:{id}'
                text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y1 - 10 - text_size[1]), (x1 + text_size[0], y1 - 10), (0, 255, 255), -1)
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)

                #Increment the counter if the bounding box toches the line:
                if y2 >= exitt_s and y2 <= exitt_e: #The person is in entry zone:
                    arrival[id] = (x1, y1)
                if id in arrival:
                    if y2 >= entry_s and y2 <= entry_e:
                        counter += 1
                        del arrival[id]

        #Show the counter regardless of scene in the video:     
        
        counter_text = f'Counter: {counter}'
        cv2.putText(frame, counter_text, counter_coor, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)

        #Show the frame:
        cv2.imshow("LOCUS 2024", frame)

        if (cv2.waitKey(30) == 27): break
        
if __name__ == "__main__":
    main()
