import cv2
import numpy as np

# Load YOLO
net = cv2.dnn.readNet(r"C:\Users\Vansh Patel\OneDrive\Desktop\Python\yolov3.weights", r"C:\Users\Vansh Patel\OneDrive\Desktop\Python\yolov3.cfg")
classes = []
with open(r"C:\Users\Vansh Patel\OneDrive\Desktop\Python\coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Define the class ID for person
person_class_id = classes.index('person')

# Open webcam
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Frame processing variables
process_every_n_frames = 1
frame_counter = 0

while True:
    ret, frame = cam.read()
    if not ret:
        break

    frame_counter += 1
    if frame_counter % process_every_n_frames != 0:
        continue

    frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
    height, width, _ = frame.shape

    # Preprocess frame
    blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)

    # Run inference
    output_layers_names = net.getUnconnectedOutLayersNames()
    layer_outputs = net.forward(output_layers_names)

    # Process detections
    boxes = []
    confidences = []
    class_ids = []
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if class_id == person_class_id and confidence > 0.5:  # Only detect persons with confidence > 0.5
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-max suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes and labels
    font = cv2.FONT_HERSHEY_PLAIN
    colors = 225,225,225    #np.random.uniform(0, 255, size=(len(classes), 3))
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = classes[class_ids[i]]
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 5), font, 1, color, 2)

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Display output
    cv2.imshow("YOLO Object Detection (Persons only)", frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
