import cv2
import numpy as np
import json
import os

# Load dictionary ArUco
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters()

# ID marker yang digunakan
REFERENCE_MARKERS = [0, 1, 2, 3]  # 4 marker: kiri atas, kanan atas, kiri bawah, kanan bawah
MARKER_REAL_DISTANCE_CM = 150  # Jarak asli antara dua marker referensi (misal antara atas kiri dan atas kanan)

# Path untuk menyimpan hasil kalibrasi
calibration_file = "data/calibration_data.json"

# Buka kamera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Konversi ke grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Deteksi marker
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        # Simpan posisi semua marker yang terdeteksi
        marker_positions = {}
        for i in range(len(ids)):
            marker_id = ids[i][0]
            if marker_id in REFERENCE_MARKERS:
                marker_positions[marker_id] = np.mean(corners[i][0], axis=0)

        # Jika minimal 2 marker referensi ditemukan, hitung skala
        if len(marker_positions) >= 2:
            if 0 in marker_positions and 1 in marker_positions:  # Prioritas: Marker atas
                x1, y1 = marker_positions[0]
                x2, y2 = marker_positions[1]
            elif 2 in marker_positions and 3 in marker_positions:  # Alternatif: Marker bawah
                x1, y1 = marker_positions[2]
                x2, y2 = marker_positions[3]
            else:
                x1, y1, x2, y2 = None, None, None, None  # Jika tidak cukup marker, jangan hitung

            if x1 is not None and x2 is not None:
                # Hitung jarak dalam piksel
                pixel_distance = np.linalg.norm([x2 - x1, y2 - y1])

                # Hitung skala cm per piksel
                cm_per_pixel = MARKER_REAL_DISTANCE_CM / pixel_distance

                # Simpan hasil kalibrasi ke file JSON
                os.makedirs(os.path.dirname(calibration_file), exist_ok=True)
                with open(calibration_file, "w") as f:
                    json.dump({"cm_per_pixel": float(cm_per_pixel)}, f)


                # Tampilkan skala di layar
                cv2.putText(frame, f'Scale: {cm_per_pixel:.4f} cm/pixel',
                            (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Tampilkan hasil
    cv2.imshow("ArUco Scale Calibration", frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
