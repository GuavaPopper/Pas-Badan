import cv2
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

def measure_body_parts():
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

        body_parts = {"Chest": [], "Shoulder": [], "Waist": [], "Hip": [], "Leg": []}

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if y < frame.shape[0] * 0.2:
                body_parts["Shoulder"].append((x, y))
            elif y < frame.shape[0] * 0.4:
                body_parts["Chest"].append((x, y))
            elif y < frame.shape[0] * 0.6:
                body_parts["Waist"].append((x, y))
            elif y < frame.shape[0] * 0.8:
                body_parts["Hip"].append((x, y))
            else:
                body_parts["Leg"].append((x, y))

        cv2.imshow("Body Measurement", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    measure_body_parts()
