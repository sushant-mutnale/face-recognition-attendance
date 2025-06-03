import cv2
import os
import time
import pandas as pd
from datetime import datetime
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

from modules.embedding_utils import get_embedding, load_embeddings, save_embeddings

def recognize_and_mark():
    embeddings, labels = load_embeddings()
    if not embeddings:
        print("❌ No face data found. Please add a face first.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam.")
        return

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    attendance_file = "attendance.csv"
    if os.path.exists(attendance_file) and os.path.getsize(attendance_file) > 0:
        df = pd.read_csv(attendance_file)
    else:
        df = pd.DataFrame(columns=["Name", "Date", "Time", "Status"])

    print("Starting webcam for attendance. Hold your face steady...")
    time.sleep(4)

    matched = False
    name = "Unknown"
    start_time = time.time()
    duration = 10  # seconds

    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            new_embedding = get_embedding(pil_img)

            if new_embedding is not None:
                sims = cosine_similarity([new_embedding], embeddings)[0]
                max_sim = max(sims)
                max_index = sims.argmax()

                if max_sim > 0.6:
                    name = labels[max_index]
                    date_today = datetime.now().strftime("%Y-%m-%d")
                    time_now = datetime.now().strftime("%H:%M:%S")

                    already_marked = df[(df["Name"] == name) & (df["Date"] == date_today)]
                    if already_marked.empty:
                        df.loc[len(df)] = [name, date_today, time_now, "Present"]
                        df.to_csv(attendance_file, index=False)
                        with open("status.txt", "w") as f:
                            f.write(f"Attendance marked for {name}")
                    else:
                        with open("status.txt", "w") as f:
                            f.write(f"Attendance already marked for {name}")
                    matched = True
                    break

        # Draw face + name (if matched)
        

        cv2.imshow("Mark Attendance - Press Q to Quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if matched:
            display_start = time.time()
            while time.time() - display_start < 4:  # show for 4 seconds
                ret, frame = cap.read()
                if not ret:
                    break

                faces = face_cascade.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 1.3, 5)
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                cv2.imshow("Mark Attendance - Press Q to Quit", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        break


    cap.release()
    cv2.destroyAllWindows()
    if not matched:
        with open("status.txt", "w") as f:
            f.write("No match found. Try again.")

if __name__ == "__main__":
    recognize_and_mark()
