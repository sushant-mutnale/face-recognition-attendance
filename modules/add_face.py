import cv2
import os
import sys
import time
from PIL import Image
from embedding_utils import get_embedding, load_embeddings, save_embeddings

def add_face(name):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam.")
        return

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    embeddings, labels = load_embeddings()
    collected = 0
    max_images = 15
    delay=0

    print(f"[INFO] Starting camera. Capturing {max_images} face samples for {name}...")
    time.sleep(6)

    while collected < max_images:

        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if not delay%3:
            cv2.imshow('frame', frame)
            delay+=1
            continue

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            embedding = get_embedding(pil_img)

            if embedding is not None:
                embeddings.append(embedding)
                labels.append(name.lower().strip())
                collected += 1
                print(f"[✓] Captured image {collected}/{max_images}")
                time.sleep(0.3)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame,f"Samples: {collected}/15",(x, y - 10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0, 0, 255),2)

        cv2.imshow("Adding Face - Press Q to Quit", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    save_embeddings(embeddings, labels)

    with open("status.txt", "w") as f:
         f.write(f"{collected} faces samples saved for '{name}'.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Name is required. Usage: python add_face.py <name>")
        sys.exit(1)

    name = sys.argv[1]
    add_face(name)
