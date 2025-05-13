import yt_dlp
import speech_recognition as sr
from googletrans import Translator
from pydub import AudioSegment
from moviepy.editor import VideoFileClip, AudioFileClip
from gtts import gTTS
import os
import whisper

def download_youtube_video_720p(url, save_path='.'):
    ydl_opts = {
        'format': 'bestvideo[height<=720]+bestaudio/best',  
        'outtmpl': f'{save_path}/%(title)s.%(ext)s',  
        'noplaylist': True,  
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        print("Download completed successfully!")
    except Exception as e:
        print(f"An error occurred during download: {e}")

def extract_audio(video_path, audio_path):
    """Extract audio from the downloaded video."""
    video = VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(audio_path)
    audio.close()
    video.close()
    print(f"Audio extracted successfully from {video_path}!")

def transcribe_audio(audio_path):
    """Transcribe audio to text using Speech Recognition."""
    audio_text = ""

    audio = AudioSegment.from_mp3(audio_path)
    audio.export('converted_audio.wav', format='wav')

    model = whisper.load_model("base")  
    
    from google import genai

    client = genai.Client(api_key="AIzaSyAF-Pq4AZdOicHKsoJfuP9ClFmUGLnCVE4")

    myfile = client.files.upload(path=audio_path)

    response = client.models.generate_content(
        model="gemini-2.0-flash", contents=["audio file may be hindi,telugu,or any language and translate into ''''english only language''' only and dont remove any words no unwanted words important i need same size of video length text", myfile]
    )

    print(response.text)
    

    return response.text

def translate_and_convert_to_tamil(audio_text, output_audio_path):
    """Translate text to Tamil and convert it to Tamil audio."""
    translator = Translator()
    
    translated_text = translator.translate(audio_text, dest='ta').text
    
    tts = gTTS(translated_text, lang='ta')
    tts.save(output_audio_path)
    print(f"Tamil audio generated successfully and saved as {output_audio_path}!")

def replace_audio_in_video(video_path, new_audio_path, output_video_path):
    """Replace the audio of the video with new audio."""
    video = VideoFileClip(video_path)
    new_audio = AudioFileClip(new_audio_path)
    
    final_video = video.set_audio(new_audio)
    final_video.write_videofile(output_video_path, codec='libx264', audio_codec='aac')
    
    video.close()
    new_audio.close()
    final_video.close()
    print(f"Replaced audio in video and saved as {output_video_path}!")

def process_videos(input_folder, output_folder):
    """Process all videos in the input folder."""
    video_files = [f for f in os.listdir(input_folder) if f.endswith(('.mp4', '.mkv', '.avi', '.mov'))]
    
    for video_file in video_files:
        video_path = os.path.join(input_folder, video_file)
        audio_output_path = os.path.join(input_folder, 'extracted_audio.mp3')
        
        extract_audio(video_path, audio_output_path)

        audio_text = transcribe_audio(audio_output_path)

        tamil_audio_output_path = os.path.join(output_folder, f'tamil_audio_{os.path.splitext(video_file)[0]}.mp3')
        translate_and_convert_to_tamil(audio_text, tamil_audio_output_path)

        output_video_path = os.path.join(output_folder, f'video_with_tamil_audio_{os.path.splitext(video_file)[0]}.mp4')
        replace_audio_in_video(video_path, tamil_audio_output_path, output_video_path)

        # Step 5: Clean up original translated audio and extracted audio
        if os.path.exists(audio_output_path):
            os.remove(audio_output_path)
            os.remove(video_path)
            print(f"Deleted extracted audio: {audio_output_path}")
        if os.path.exists(tamil_audio_output_path):
            os.remove(tamil_audio_output_path)
            print(f"Deleted translated audio: {tamil_audio_output_path}")


def ad(video_url):
    video_url = video_url  
    download_youtube_video_720p(video_url, save_path=r'C:\Users\deepak\Downloads\codeeee\codeeee\download')  
    input_folder = r'C:\Users\deepak\Downloads\codeeee\codeeee\download'  
    output_folder = r'C:\Users\deepak\Downloads\codeeee\codeeee\translated_audio' 
    os.makedirs(output_folder, exist_ok=True)
    process_videos(input_folder, output_folder)