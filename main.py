import aruco_detection
import height_measurement
import body_part_measurement

if __name__ == "__main__":
    print("Mulai kalibrasi...")
    aruco_detection.calibrate_scale()

    print("Mengukur tinggi badan...")
    height_measurement.measure_height()

    print("Mengukur bagian tubuh...")
    body_part_measurement.measure_body_parts()
