import os
import requests
from gtts import gTTS
from moviepy.editor import *
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
import cv2  # OpenCV for image handling
import numpy as np
import textwrap
import streamlit as st
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips

# Load environment variables
load_dotenv()
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Create folders
os.makedirs("assets/images", exist_ok=True)
os.makedirs("assets/audio", exist_ok=True)

def generate_script(topic):
    model = ChatGroq(
        temperature=0.7,
        groq_api_key=GROQ_API_KEY,
        model_name="llama3-70b-8192"
    )

    prompt = PromptTemplate.from_template("""
    ###INSTRUCTIONS
    Generate only facts no preamble or anything.
    Use a friendly, energetic, and clear tone. Make sure the pacing fits within 60 seconds, with natural pauses between each fact about the following topic:
    {topic}
    Do not provide preamble.
    ###Facts(NO PREAMBLE):
    """)

    chain = prompt | model
    response = chain.invoke({"topic": topic})
    lines = response.content.strip().split('\n')
    return '\n'.join(lines[1:]) if len(lines) > 1 else response.content.strip()

def fetch_images(query, num_images=5):
    headers = {"Authorization": PEXELS_API_KEY}
    url = f"https://api.pexels.com/v1/search?query={query}&per_page={num_images}"
    res = requests.get(url, headers=headers)

    if res.status_code != 200:
        raise Exception(f"Pexels API error: {res.text}")

    data = res.json()
    if "photos" not in data or not data["photos"]:
        raise Exception("No images found.")

    image_paths = []
    for i, photo in enumerate(data['photos']):
        img_url = photo['src']['portrait']
        img_path = f"assets/images/img_{i}.jpg"
        img_data = requests.get(img_url).content
        with open(img_path, 'wb') as f:
            f.write(img_data)
        image_paths.append(img_path)
    return image_paths

def generate_voice(script_text, output_path="assets/audio/voice.mp3"):
    tts = gTTS(text=script_text, lang="hi")
    tts.save(output_path)
    return output_path

def split_script_evenly(script, parts):
    words = script.strip().split()
    chunk_size = len(words) // parts
    chunks = [' '.join(words[i * chunk_size : (i + 1) * chunk_size]) for i in range(parts - 1)]
    chunks.append(' '.join(words[(parts - 1) * chunk_size:]))
    return chunks

def wrap_text(text, width=40):
    words = text.split()
    lines, line = [], []
    for word in words:
        if len(' '.join(line + [word])) <= width:
            line.append(word)
        else:
            lines.append(' '.join(line))
            line = [word]
    lines.append(' '.join(line))
    return lines

def create_video(image_paths, script_text, voice_path):
    script_chunks = split_script_evenly(script_text, len(image_paths))
    audio = AudioFileClip(voice_path)
    total_duration = audio.duration
    per_image_duration = total_duration / len(image_paths)

    clips = []

    for i, (img_path, text) in enumerate(zip(image_paths, script_chunks)):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (1080, 1920))  # standard 9:16

        wrapped_lines = wrap_text(text, width=35)
        y0 = 1600
        for j, line in enumerate(wrapped_lines):
            y = y0 + j * 60
            cv2.putText(
                img, line, (60, y),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                (255, 255, 255), 3, cv2.LINE_AA
            )

        temp_path = f"assets/images/temp_frame_{i}.jpg"
        cv2.imwrite(temp_path, img)

        clip = ImageClip(temp_path).set_duration(per_image_duration)
        clips.append(clip)

    final_video = concatenate_videoclips(clips).set_audio(audio).set_fps(30)
    output_path = "youtube_short.mp4"
    final_video.write_videofile(output_path, codec="libx264", audio_codec="aac")
    return output_path

# Streamlit UI
st.title("🎬 YouTube Shorts Generator from Topic")
topic = st.text_input("Enter a topic (Like., Parasuram, Galaxy, Oceans)")

if st.button("Generate YouTube Short"):
    if topic:
        try:
            st.info("Generating script...")
            script = generate_script(topic)
            st.success("script generated successfully!")

            st.info("Fetching images...")
            images = fetch_images(topic)

            st.info("Generating voiceover...")
            voice_path = generate_voice(script)

            st.info("Creating video...")
            video_path = create_video(images, script, voice_path)

            st.success("Video created successfully!")
            st.video(video_path)

        except Exception as e:
            st.error(f"something went wrong: {e}")
    else:
        st.warning("Please enter a topic first.")
