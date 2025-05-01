import os
import requests
from gtts import gTTS
from moviepy.editor import *
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from nlp import tokenize  # Make sure this function just extracts clean topic
from PIL import Image, ImageDraw, ImageFont
import textwrap
import streamlit as st

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
    tts = gTTS(text=script_text, lang="en")
    tts.save(output_path)
    return output_path

def create_video(image_paths, script_text, voice_path):
    lines = script_text.split('.')[:len(image_paths)]
    clips = []

    font_path = "./calibri.ttf"  # Replace with a valid TTF path
    font_size = 48

    for i, (img_path, line) in enumerate(zip(image_paths, lines)):
        line = line.strip()
        if not line:
            continue

        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(font_path, font_size)
        wrapped_text = textwrap.fill(line, width=40)
        text_size = draw.textbbox((0, 0), wrapped_text, font=font)
        x = (img.width - text_size[2]) // 2
        y = img.height - 250

        draw.rectangle([(x - 20, y - 20), (x + text_size[2] + 20, y + text_size[3] + 20)], fill=(0, 0, 0, 150))
        draw.text((x, y), wrapped_text, font=font, fill=(255, 255, 255))

        temp_img_path = f"assets/images/temp_img_{i}.jpg"
        img.save(temp_img_path)

        clip = ImageClip(temp_img_path).set_duration(3).resize(height=1920).set_position("center")
        clips.append(clip)

    audio = AudioFileClip(voice_path)
    final_video = concatenate_videoclips(clips).set_audio(audio).set_fps(30)
    output_path = "youtube_short.mp4"
    final_video.write_videofile(output_path, codec='libx264', audio_codec='aac')
    return output_path

# Streamlit UI
st.title("🎬 YouTube Shorts Generator from Topic")
topic = st.text_input("Enter a topic (e.g., Parasuram, Black Holes, Oceans)")

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
