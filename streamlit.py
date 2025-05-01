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
from PIL import Image, ImageDraw, ImageFont

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
def draw_centered_hindi_text(img_path, text, output_path, font_path, font_size=60):
    # Open image with PIL
    img = Image.open(img_path).convert("RGB")
    img = img.resize((1080, 1920))
    
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, font_size, encoding="unic")

    # Wrap text
    lines = textwrap.wrap(text, width=25)
    text_height = sum([draw.textbbox((0, 0), line, font=font)[3] for line in lines])
    y = (img.height - text_height) // 2

    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        text_width = bbox[2] - bbox[0]
        x = (img.width - text_width) // 2
        draw.text((x, y), line, font=font, fill=(255, 255, 255))
        y += bbox[3] + 10

    img.save(output_path)

def create_video(image_paths, script_text, voice_path):
    script_chunks = split_script_evenly(script_text, len(image_paths))
    audio = AudioFileClip(voice_path)
    total_duration = audio.duration
    per_image_duration = total_duration / len(image_paths)
    font_path = "fonts/NotoSansDevanagari-Regular.ttf"  # Replace with your path

    clips = []

    for i, (img_path, text) in enumerate(zip(image_paths, script_chunks)):
        temp_img_path = f"assets/images/temp_hindi_img_{i}.jpg"
        draw_centered_hindi_text(img_path, text, temp_img_path, font_path)

        clip = ImageClip(temp_img_path).set_duration(per_image_duration)
        clips.append(clip)

    final_video = concatenate_videoclips(clips).set_audio(audio).set_fps(30)
    output_path = "youtube_short.mp4"
    final_video.write_videofile(output_path, codec="libx264", audio_codec="aac")
    return output_path

# Streamlit UI
st.title("🎬 YouTube Shorts Generator from Topic\n")
topic = st.text_input("Enter a topic (Like., Parasuram, Black Holes, Oceans)")

if st.button("Generate YouTube Short"):
    if topic:
        try:
            st.info("Generating script...")
            script = generate_script(topic)
            st.success("Script generated successfully!")

            st.info("Fetching images...")
            images = fetch_images(topic)

            st.info("Generating voiceover...")
            voice_path = generate_voice(script)

            st.info("Creating video...")
            video_path = create_video(images, script, voice_path)

            st.success("Video created successfully!")
            st.video(video_path)

        except Exception as e:
            st.error(f"Something went wrong: {e}")
    else:
        st.warning("Please enter a topic first.")
