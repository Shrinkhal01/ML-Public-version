import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import os

def notify_authorities(subject, body, attachments):
    # Email configuration
    sender_email = "shrinkhalshrinkhal@gmail.com"  # Replace with your email
    receiver_emails = ["privatehvro@gmail.com", "shrinkhalshrinkhal22@gmail.com", "adityabhatt78910@gmail.com", "lakshitatak1@gmail.com", "iishasharrma@gmail.com"]
    password = "pptsbbotjoqgwdkp"  # The password for the sender email (App Password)

    # Create the email
    msg = MIMEMultipart() # this is used to create a multipart email
    msg['From'] = sender_email
    msg['To'] = ", ".join(receiver_emails)
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    # Attach files
    for file_path in attachments:
        try:
            with open(file_path, "rb") as attachment:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(attachment.read())
            encoders.encode_base64(part)
            part.add_header(
                "Content-Disposition",
                f"attachment; filename={os.path.basename(file_path)}",
            )
            msg.attach(part)
        except Exception as e:
            print(f"Failed to attach {file_path}: {e}")

    try:
        # Connect to the email server and send the email
        server = smtplib.SMTP('smtp.gmail.com', 587)  # Use the appropriate SMTP server
        server.starttls()
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_emails, msg.as_string())
        server.quit()
        print("Notification sent to authorities.")
    except Exception as e:
        print(f"Failed to send notification: {e}")