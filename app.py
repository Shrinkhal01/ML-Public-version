# app.py

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import img_to_array
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import smtplib
import gradio as gr

# ── Configuration via ENV variables ─────────────────────────────────────────────
MODEL_PATH       = os.getenv("MODEL_PATH", "my_model.keras")
TRAIN_DIR        = os.getenv("TRAIN_DIR", "train")
SCREENSHOT_DIR   = os.getenv("SCREENSHOT_DIR", "screenshots")
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

    sender_email = "shrinkhalshrinkhal@gmail.com"  # Replace with your email
    receiver_emails = ["privatehvro@gmail.com", "shrinkhalshrinkhal22@gmail.com", "adityabhatt78910@gmail.com", "lakshitatak1@gmail.com", "iishasharrma@gmail.com","idforextraapp@gmail.com"]
    password = "pptsbbotjoqgwdkp"  # The password for the sender email (App Password)


SENDER_EMAIL     = os.getenv("your email")
EMAIL_PASSWORD   = os.getenv("your email password")
RECEIVER_EMAILS  = os.getenv("shrinkhalshrinkhal22@gmail.com","adityabhatt78910@gmail.com","lakshitatak1@gmail.com","iishasharrma@gmail.com","idforextraapp@gmail.com","").split(",")
CONFIDENCE_THRESH= float(os.getenv("CONF_THRESH", "0.85"))
MAX_EMAILS       = int(os.getenv("MAX_EMAILS", "2"))
# ──────────────────────────────────────────────────────────────────────────────

def get_class_names(train_dir):
    return sorted([d for d in os.listdir(train_dir) if not d.startswith('.')])

def notify_authorities(subject, body, attachments):
    msg = MIMEMultipart()
    msg["From"], msg["To"], msg["Subject"] = SENDER_EMAIL, ", ".join(RECEIVER_EMAILS), subject
    msg.attach(MIMEText(body, "plain"))
    for path in attachments:
        with open(path, "rb") as f:
            part = MIMEBase("application","octet-stream")
            part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f'attachment; filename="{os.path.basename(path)}"')
        msg.attach(part)
    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    server.login(SENDER_EMAIL, EMAIL_PASSWORD)
    server.sendmail(SENDER_EMAIL, RECEIVER_EMAILS, msg.as_string())
    server.quit()

def detect_accident_frame(frame, model, class_names):
    x = cv2.resize(frame, (1280,720))
    x = img_to_array(x)[None,...]
    preds = model.predict(x)[0]
    idx   = int(np.argmax(preds))
    conf  = float(preds[idx])
    name  = class_names[idx]
    return name, conf

# Gradio inference function
def run_detector(video_file):
    model       = tf.keras.models.load_model(MODEL_PATH)
    class_names = get_class_names(TRAIN_DIR)

    cap = cv2.VideoCapture(video_file.name)
    acc_cnt = 0
    acc_frames = []
    emails_sent = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        name, conf = detect_accident_frame(frame, model, class_names)
        if name.lower()=="accident" and conf>=CONFIDENCE_THRESH:
            acc_cnt += 1
            # red border
            cv2.rectangle(frame,(0,0),(frame.shape[1],frame.shape[0]),(0,0,255),10)
            path = os.path.join(SCREENSHOT_DIR, f"acc_{acc_cnt}.jpg")
            cv2.imwrite(path, frame)
            acc_frames.append(path)

            if acc_cnt>=4 and emails_sent<MAX_EMAILS:
                notify_authorities(
                    subject="Accident Detected!",
                    body="Accident alert—see attachments.",
                    attachments=acc_frames
                )
                emails_sent += 1
                acc_cnt=0
                acc_frames=[]
        else:
            acc_cnt, acc_frames = 0, []

    cap.release()
    return f"Done processing. Emails sent: {emails_sent}", None

# Gradio UI
iface = gr.Interface(
    fn=run_detector,
    inputs=gr.File(label="Upload MP4"),
    outputs=gr.Textbox(label="Status"),
    title="Accident Detector + Notifier"
)

if __name__=="__main__":
    iface.launch()
