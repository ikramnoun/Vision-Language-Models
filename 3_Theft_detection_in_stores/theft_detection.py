import cv2
import numpy as np
import os
import time
import base64
import threading
import os
from openai import OpenAI


# Set up OpenAI client for OpenRouter.ai
OPENROUTER_API_KEY = ""  # Replace with your OpenRouter API key
client = OpenAI(
    base_url="https://openrouter.ai/api/v1/chat/completions",
    api_key=OPENROUTER_API_KEY,
)

# Create folder for saving full-frame images
FRAME_FOLDER = "full_frames"
os.makedirs(FRAME_FOLDER, exist_ok=True)

# Output text file
OUTPUT_FILE = "observations.txt"

# Timer settings
last_sent_time = 0
SEND_INTERVAL = 4  # Send an image every 4 seconds


def analyze_with_ai_model(image_path, timestamp):
    """Sends the full frame to ai_model via OpenRouter.ai to check for money theft at the cash counter."""
    try:
        with open(image_path, "rb") as img_file:
            base64_image = base64.b64encode(img_file.read()).decode("utf-8")

        # Prepare the message for the AI
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """
                        Observe the **cash counter area** and respond in a structured format.
                        If money theft is detected (**"Yes"**), provide details of the **suspect**.
                        NO More Details  
                        | Suspicious Activity at Cash Counter | Observed? (Yes/No) | Suspect Description (If Yes) |
                        |--------------------------------------|--------------------|-----------------------------|
                        | Money theft from cash counter?      |                    |                             |

                        If theft is detected, describe the **clothing, appearance, and any identifiable features** of the suspect.
                        Otherwise, leave the details column empty.
                        """
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]

        # Call the OpenAI API via OpenRouter.ai
        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "<YOUR_SITE_URL>",  # Optional. Replace with your site URL.
                "X-Title": "<YOUR_SITE_NAME>",  # Optional. Replace with your site name.
            },
            extra_body={},
            model="openrouter/free",  # Use the ai_model 
            messages=messages,
        )

        observation = completion.choices[0].message.content.strip()

        # Only save observations where theft is confirmed
        if "Yes" in observation:
            with open(OUTPUT_FILE, "a", encoding="utf-8") as file:
                file.write(f"{timestamp} - {observation}\n")
            print(f"✅ Observation Saved: {observation}")

    except Exception as e:
        print(f"❌ Error analyzing image: {e}")


def process_frame(frame):
    """Saves the full frame and starts a thread for AI analysis every 4 seconds."""
    global last_sent_time

    if frame is None or frame.size == 0:
        print("⚠️ Warning: Empty frame received, skipping...")
        return

    current_time = time.time()
    if current_time - last_sent_time >= SEND_INTERVAL:
        last_sent_time = current_time

        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        image_filename = os.path.join(FRAME_FOLDER, f"frame_{timestamp}.jpg")
        cv2.imwrite(image_filename, frame)

        # Run ai_model AI analysis in a separate thread
        ai_thread = threading.Thread(target=analyze_with_ai_model, args=(image_filename, timestamp))
        ai_thread.daemon = True
        ai_thread.start()


def start_monitoring(video_file):
    """Reads video frames and monitors the cash counter area."""
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print("❌ Error: Could not open video file.")
        return

    cv2.namedWindow("Cash Counter Monitoring")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (1020, 500))  # Resize for better display
        process_frame(frame)

        cv2.imshow("Cash Counter Monitoring", frame)

        if cv2.waitKey(30) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Monitoring Completed.")


if __name__ == "__main__":
    video_file = "susp.mp4"
    start_monitoring(video_file)
