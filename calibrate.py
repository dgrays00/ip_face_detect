import cv2

IP_URL = "http://10.18.178.243:8080/video"  # kendi IP'ni yaz

cap = cv2.VideoCapture(IP_URL)
if not cap.isOpened():
    print("IP kamera acilamadi!")
    raise SystemExit

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

print("1 metre uzaga gec. Yuz kutusu genisligini (w) konsolda goreceksin. ESC ile cik.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))

    if len(faces) > 0:
        # en buyuk yuzu sec
        x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, f"face_w_px={w}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        print("face_w_px =", w)

    cv2.imshow("Calibrate", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
