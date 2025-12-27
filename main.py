import cv2
import time

IP_URL = "http://10.18.178.243:8080/video"

cap = cv2.VideoCapture(IP_URL)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
if not cap.isOpened():
    print("IP kamera acilamadi!")
    raise SystemExit

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# 👉 Baslangicta icerde 15 kisi var
iceride = 15
giren = 0
cikan = 0

# ⚡ Performans
TARGET_WIDTH = 640
DETECT_EVERY = 2
frame_id = 0

# 🟥 Kapı sayım karesi
BOX_W_RATIO = 0.35
BOX_H_RATIO = 0.60

# 🙂 Yüz algılama
FACE_SCALE = 1.05
FACE_NEIGHBORS = 3
FACE_MINSIZE = (30, 30)

# 📏 Yüz menzili (px) ~ 0.8–2 m
FACE_MIN_W = 45
FACE_MAX_W = 140

# 🚶 Vücut filtresi
MIN_PERSON_AREA = 22000

# 🧠 Stabilite
ENTER_FRAMES_NEEDED = 2
EXIT_FRAMES_NEEDED = 6

inside_frames = 0
outside_frames = 999
counted_this_entry = False

print("Basladi. ESC ile cikis.")

faces = []
people = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1

    # 🔽 Goruntuyu kucult
    h0, w0 = frame.shape[:2]
    new_h = int(h0 * (TARGET_WIDTH / w0))
    frame = cv2.resize(frame, (TARGET_WIDTH, new_h))

    h, w = frame.shape[:2]

    # 🟥 Orta kare
    box_w = int(w * BOX_W_RATIO)
    box_h = int(h * BOX_H_RATIO)
    box_x1 = int(w/2 - box_w/2)
    box_y1 = int(h/2 - box_h/2)
    box_x2 = box_x1 + box_w
    box_y2 = box_y1 + box_h

    # Algılama
    if frame_id % DETECT_EVERY == 0:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=FACE_SCALE,
            minNeighbors=FACE_NEIGHBORS, minSize=FACE_MINSIZE
        )

        people, _ = hog.detectMultiScale(
            frame, winStride=(8, 8), padding=(8, 8), scale=1.05
        )

    # 🟥 Kare içindekiler
    faces_in_box = []
    people_in_box = []

    for (x, y, wf, hf) in faces:
        cx, cy = x + wf//2, y + hf//2
        if box_x1 < cx < box_x2 and box_y1 < cy < box_y2:
            faces_in_box.append((x, y, wf, hf))

    for (x, y, wp, hp) in people:
        cx, cy = x + wp//2, y + hp//2
        if box_x1 < cx < box_x2 and box_y1 < cy < box_y2:
            if (wp * hp) >= MIN_PERSON_AREA:
                people_in_box.append((x, y, wp, hp))

    # 🙂 Geçerli yüz var mı? (menzile göre)
    valid_face = None
    for (x, y, wf, hf) in faces_in_box:
        if FACE_MIN_W <= wf <= FACE_MAX_W:
            valid_face = (x, y, wf, hf)
            break

    has_valid_face = valid_face is not None
    has_person = len(people_in_box) > 0

    in_box = has_person  # karede vücut varsa içeride kabul

    # 🧠 Stabil
    if in_box:
        inside_frames += 1
        outside_frames = 0
    else:
        outside_frames += 1
        inside_frames = 0

    if outside_frames >= EXIT_FRAMES_NEEDED:
        counted_this_entry = False

    # ✅ SAYMA
    if (not counted_this_entry) and inside_frames >= ENTER_FRAMES_NEEDED:
        if has_person and has_valid_face:
            giren += 1
            iceride += 1
            counted_this_entry = True
            print("GIREN sayildi")
        elif has_person and (not has_valid_face) and iceride > 0:
            cikan += 1
            iceride -= 1
            counted_this_entry = True
            print("CIKAN sayildi")

    # ==== CIZIMLER ====
    cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (255, 255, 0), 2)

    for (x, y, wp, hp) in people_in_box:
        cv2.rectangle(frame, (x, y), (x+wp, y+hp), (0, 255, 0), 2)

    if valid_face:
        x, y, wf, hf = valid_face
        cv2.rectangle(frame, (x, y), (x+wf, y+hf), (255, 0, 0), 2)

    cv2.putText(frame, f"Giren: {giren}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(frame, f"Cikan: {cikan}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    cv2.putText(frame, f"Iceride: {iceride}", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

    cv2.imshow("Kapi Giris-Cikis Sayma", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
