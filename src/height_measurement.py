import cv2
import numpy as np
import json
from color_thresholding import detect_green_pin

calibration_file = "data/calibration_data.json"

def load_scale():
    try:
        with open(calibration_file, "r") as f:
            data = json.load(f)
            return data["cm_per_pixel"]
    except FileNotFoundError:
        print("Kalibrasi belum dilakukan!")
        return None

def measure_height():
    cm_per_pixel = load_scale()
    if cm_per_pixel is None:
        return

    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        mask, result = detect_green_pin(frame)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            y_values = [cv2.boundingRect(cnt)[1] for cnt in contours]
            min_y, max_y = min(y_values), max(y_values)

            pixel_height = max_y - min_y
            real_height = pixel_height * cm_per_pixel

            cv2.putText(frame, f'Height: {real_height:.2f} cm', (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Height Measurement", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    measure_height()
