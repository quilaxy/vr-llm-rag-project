import os
from os import PathLike
from time import time
import asyncio
from typing import Union
import uuid

from dotenv import load_dotenv
from google.cloud import texttospeech
from deepgram import Deepgram
import pygame
from pygame import mixer
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from record import speech_to_text

# Load API keys
load_dotenv()
OPENAI_API_KEY = os.getenv("LLM_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

# Initialize APIs
deepgram = Deepgram(DEEPGRAM_API_KEY)
mixer.init()

# Inisialisasi Google Text-to-Speech Client
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "D:/ITS/Semester 7/Protel/vr-llm-rag/service_tts.json"
google_tts_client = texttospeech.TextToSpeechClient()

# Inisialisasi OpenAI LangChain
llm = ChatOpenAI(
    api_key=os.getenv("LLM_API_KEY"), 
    model="gpt-4o-mini", 
    temperature=0.7
)

context = "Kamu adalah Nathan, seorang ahli sejarah Indonesia dengan kepribadian ceria dan bersemangat. Kamu berbicara dengan nada ramah dan menarik, memberikan penjelasan singkat yang mudah dipahami. Jawabanmu harus singkat, maksimal 1-2 kalimat. Beri jeda antar kalimat sebanyak 1 detik."
RECORDING_PATH = "audio/recording.wav"

# Konversi teks menjadi audio WAV menggunakan Google Text-to-Speech
def text_to_speech_file(text: str, filename: str) -> str:
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code="id-ID",
        ssml_gender=texttospeech.SsmlVoiceGender.MALE
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16  # WAV format
    )

    # Panggil Google Text-to-Speech API
    response = google_tts_client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    # Simpan audio sebagai file WAV
    save_file_path = f"audio/{filename}"
    with open(save_file_path, "wb") as out:
        out.write(response.audio_content)
    # print(f"Audio berhasil disimpan ke {save_file_path}")
    
    return save_file_path

def request_gpt(prompt: str) -> str:
    messages = [HumanMessage(content=prompt)]
    response = llm.generate(messages=[messages])
    return response.generations[0][0].text

async def transcribe(file_name: Union[Union[str, bytes, PathLike[str], PathLike[bytes]], int]):
    with open(file_name, "rb") as audio:
        source = {"buffer": audio, "mimetype": "audio/wav"}
        response = await deepgram.transcription.prerecorded(source, {'language': 'id'})
        return response["results"]["channels"][0]["alternatives"][0]["transcript"]

def log(log: str):
    print(log)
    with open("status.txt", "w") as f:
        f.write(log)

def introduction():
    intro_text = """
    Halo! Saya Nathan, teman diskusi kamu hari ini. Saya di sini untuk membantu kamu belajar tentang sejarah Indonesia. 
    Kamu mau belajar tentang apa hari ini?
    """
    
    intro_file_path = text_to_speech_file(intro_text, "intro.wav")
    
    # Mainkan audio pengenalan
    mixer.Sound(intro_file_path).play()
    pygame.time.wait(int(mixer.Sound(intro_file_path).get_length() * 1000))
    print(intro_text)

if __name__ == "__main__":
    introduction()
    while True:
        # Rekam audio
        log("Mendengarkan...")
        speech_to_text()
        log("Selesai mendengarkan")

        # Transkripsi audio
        current_time = time()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        transcript = loop.run_until_complete(transcribe(RECORDING_PATH))
        transcription_time = time() - current_time
        log(f"Transkripsi selesai dalam {transcription_time:.2f} detik.")
        print(f"Transkripsi: {transcript}")

        # Dapatkan respon dari GPT
        context += f"\nPengguna: {transcript}\nNathan: "
        response = request_gpt(context)
        context += response

        # Konversi respons menjadi audio
        response_audio_file = text_to_speech_file(response, "response.wav")
        sound = mixer.Sound(response_audio_file)
        with open("conv.txt", "a") as f:
            f.write(f"{response}\n")
        sound.play()
        pygame.time.wait(int(sound.get_length() * 1000))
        
        # print(f"\nUser: {transcript}\nAI: {response}")
        print(f"\nAI: {response}")
