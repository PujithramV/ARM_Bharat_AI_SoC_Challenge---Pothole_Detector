import cv2
import numpy as np
import time
import tflite_runtime.interpreter as tflite

# --- CONFIGURATION ---
MODEL_PATH = "best-int8.tflite" 
CONF_THRESHOLD = 0.45  # Minimum confidence to show a detection
IOU_THRESHOLD = 0.45   # Threshold for removing overlapping boxes

# --- INITIALIZE AI ---
print("Loading Model...")
interpreter = tflite.Interpreter(model_path=MODEL_PATH, num_threads=4)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
# YOLOv5 export was set to 320x320
height, width = input_details[0]['shape'][1], input_details[0]['shape'][2]

# --- START CAMERA ---
# 0 is usually the default index for the first USB webcam (/dev/video0)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

print("Starting live feed... Press 'q' to quit.")

while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # 1. Preprocess the image for the AI (Resize and format)
    img_resized = cv2.resize(frame, (width, height))
    input_data = np.expand_dims(img_resized, axis=0)
    
    # Int8 models usually require uint8 input
    if input_details[0]['dtype'] == np.uint8:
        input_data = input_data.astype(np.uint8)
    else:
        input_data = (input_data.astype(np.float32) / 255.0)

    # 2. Run the AI Inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]

    # 3. Process the Results (Extract boxes)
    boxes = []
    scores = []
    
    # Loop through the AI's predictions
    for detection in output_data:
        # Depending on quantization, you might need to adjust scales. 
        # Assuming standard YOLOv5 TFLite flat output: [x, y, w, h, conf, class_prob]
        confidence = detection[4]
        
        # De-quantize confidence if output is uint8 or int8
        if output_details[0]['dtype'] in [np.uint8, np.int8]:
            scale, zero_point = output_details[0]['quantization']
            confidence = (confidence - zero_point) * scale

        if confidence > CONF_THRESHOLD:
            class_score = detection[5] if len(detection) > 5 else 1.0
            if output_details[0]['dtype'] in [np.uint8, np.int8]:
                 class_score = (class_score - zero_point) * scale
                 
            final_score = confidence * class_score
            
            if final_score > CONF_THRESHOLD:
                # YOLO outputs center x, center y, width, height (normalized 0-1)
                cx, cy, w, h = detection[0], detection[1], detection[2], detection[3]
                if output_details[0]['dtype'] in [np.uint8, np.int8]:
                    cx, cy, w, h = [(val - zero_point) * scale for val in [cx, cy, w, h]]
                
                boxes.append([cx - w/2, cy - h/2, w, h])
                scores.append(float(final_score))

    # 4. Filter out duplicate boxes
    indices = cv2.dnn.NMSBoxes(boxes, scores, float(CONF_THRESHOLD), float(IOU_THRESHOLD))
    
    # 5. Draw the final boxes on the screen
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            score = scores[i]
            
            # Scale coordinates back up to the 640x480 frame size
            h_scale, w_scale = frame.shape[0], frame.shape[1]
            x1, y1 = int(x * w_scale), int(y * h_scale)
            x2, y2 = int((x + w) * w_scale), int((y + h) * h_scale)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"Pothole: {int(score*100)}%", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Show FPS
    fps = 1.0 / (time.time() - start_time)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the video window
    cv2.imshow('Real Time PotHole Detection - Live', frame)
    
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()