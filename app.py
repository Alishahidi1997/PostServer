from flask import Flask, request, jsonify
from transformers import pipeline
import whisper
import cv2
import base64
import openai
from openai import OpenAI
from moviepy import VideoFileClip
import os
# Initialize the OpenAI client
client = OpenAI()

# Initialize Flask app
app = Flask(__name__)

# Load the Whisper model for transcription
model = whisper.load_model("base")

# Load the emotion classifier
emotion_classifier = pipeline("text-classification", model='bhadresh-savani/distilbert-base-uncased-emotion')


# Endpoint to process text with emotion detection
@app.route('/process-text', methods=['POST'])
def process_text():
    data = request.json
    text_input = data.get('text')

    # Perform emotion detection on the text
    emotion = emotion_classifier(text_input)
    detected_emotion = emotion[0]["label"]

    # Return both the original text and detected emotion
    return jsonify({
        'text': text_input,
        'detected_emotion': detected_emotion
    })


# Endpoint to process video frames
@app.route('/process-video', methods=['POST'])
def process_video():
    # Load and decode the video sent as a base64 string
    data = request.json
    video_base64 = data.get('video_base64')

    if not video_base64:
        return jsonify({'error': 'No video data provided'}), 400
    
    # Decode the video file
    video_bytes = base64.b64decode(video_base64)
    video_path = "./temp_video.mp4"
    
    # Save to a temporary location for processing
    with open(video_path, "wb") as video_file:
        video_file.write(video_bytes)

    audio_output_path = "./output_audio.mp3"
    try:
        ExtractVoice(video_path, audio_output_path)
        textWithEmotion = VoiceToText_DetectEmotion()
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
    # Read video frames using OpenCV
    video = cv2.VideoCapture(video_path)
    base64Frames = []
    
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
    video.release()

    print(len(base64Frames), "frames read.")

    # Select frames to send (e.g., every 50th frame)
    selected_frames = base64Frames[0::15]
    print("Selected Frames" , len(selected_frames))
    # Send selected frames to OpenAI for analysis
    prompt_messages = [
        {
            "role": "user",
            "content": [
                "You are a supportive assistant for therapists working with non-speaking autistic children. These are frames from a video that I want to upload. Generate a compelling description related to the body language of the person in the video. He is an autistic person.",
                *map(lambda x: {"image": x, "resize": 768}, selected_frames),
            ],
        },
    ]

    params = {
        "model": "gpt-4o",
        "messages": prompt_messages,
        "max_tokens": 200,
    }

    result = client.chat.completions.create(**params)
    description = result.choices[0].message.content

    return jsonify({
        'description': description,
        'text': textWithEmotion
    })


def ExtractVoice(video_path, audio_output_path="output_audio.mp3"):
    """
    Extracts the audio from a video file and saves it as an MP3 file.

    Args:
        video_path (str): Path to the input video file.
        audio_output_path (str): Path to save the extracted audio file.
    """
    # Load the video
    video = VideoFileClip(video_path)

    # Extract audio
    audio = video.audio

    # Save the audio file
    audio.write_audiofile(audio_output_path)

    print(f"Audio extracted and saved to {audio_output_path}")
    

def VoiceToText_DetectEmotion():
    audio_file= open("output_audio.mp3", "rb")
    transcription = client.audio.transcriptions.create(
    model="whisper-1", 
    file=audio_file
    )


    # Perform emotion detection on the text
    emotion = emotion_classifier(transcription.text )
    detected_emotion = emotion[0]["label"]

    # Return both the original text and detected emotion
    return transcription.text + "(tone: " + detected_emotion + ")"

# Endpoint to process single image frames
@app.route('/process-image', methods=['POST'])
def process_image():
    data = request.json
    base64_image = data.get('image_base64')

    if not base64_image:
        return jsonify({'error': 'No image data provided'}), 400

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What is in this image? You are a supportive assistant for therapists working with non-speaking autistic children.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        },
                    },
                ],
            }
        ],
    )

    print(response.choices[0].message.content)
    return jsonify({
        'text': response.choices[0].message.content,
    })


if __name__ == '__main__':
    # Get the PORT from the environment variable, or default to 5000 for local testing
    port = int(os.environ.get("PORT", 5000))
    # Bind to all network interfaces using host='0.0.0.0'
    app.run(host='0.0.0.0', port=port)
