import cv2


def detect_object():
    config_file = './tensorflow/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    frozen_model = './tensorflow/frozen_inference_graph.pb'
    model = cv2.dnn_DetectionModel(frozen_model, config_file)
    model.setInputSize(320, 320)
    model.setInputScale(1.0 / 127.5)
    model.setInputMean((127.5, 127.5, 127.5))
    model.setInputSwapRB(True)

    class_labels = []
    file_name = './tensorflow/labels.txt'
    with open(file_name, 'rt') as fpt:
        class_labels = fpt.read().rstrip('\n').split('\n')
    cap = cv2.VideoCapture('./video/video3.mp4')

    font_scale = 3
    font = cv2.FONT_HERSHEY_PLAIN

    while True:
        ret, frame = cap.read()
        class_index, confidence, bbox = model.detect(frame, confThreshold=0.55)
        if len(class_index) != 0:
            for class_ind, conf, boxes in zip(class_index.flatten(), confidence.flatten(), bbox):
                if class_ind <= 80 and (class_ind - 1) == 0:
                    x1 = boxes[0]
                    y1 = boxes[1]
                    x2 = boxes[2]
                    y2 = boxes[3]
                    cx = int(((x1 + x2) / 2))
                    cy = int(((y1 + y2) / 2))
                    cv2.rectangle(frame, (x1, y1, x2, y2), (255, 0, 0), 2)
                    cv2.circle(frame, (cx, cy), 2, (255, 255, 255), 1)

        cv2.imshow('Object detection demo', frame)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
