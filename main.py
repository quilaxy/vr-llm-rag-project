import os
from os import PathLike
from time import time
import asyncio
from typing import Union

from dotenv import load_dotenv
import openai
from deepgram import Deepgram
import pygame
from pygame import mixer
import elevenlabs

from record import speech_to_text

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

# Load API keys
load_dotenv()
OPENAI_API_KEY = os.getenv("LLM_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
elevenlabs.set_api_key(os.getenv("ELEVENLABS_API_KEY"))

# Initialize APIs
gpt_client = openai.Client(api_key=OPENAI_API_KEY)
deepgram = Deepgram(DEEPGRAM_API_KEY)
mixer.init()

context = "Kamu adalah Nathan, seorang ahli sejarah Indonesia. Kamu cerdas dan memiliki kepribadian menarik. Kamu memiliki gaya bicara yang menyenangkan. Jawabanmu harus singkat, maksimal 1-2 kalimat."
conversation = {"Conversation": []}
RECORDING_PATH = "audio/recording.wav"

# Inisialisasi OpenAI dengan LangChain
llm = ChatOpenAI(
    api_key=os.getenv("LLM_API_KEY"), 
    model="gpt-4o-mini",
    temperature=0.7
)

def request_gpt(prompt: str) -> str:

    messages = [
        HumanMessage(content=prompt)
    ]

    response = llm.generate(messages=[messages])
    
    return response.generations[0][0].text

async def transcribe(file_name: Union[Union[str, bytes, PathLike[str], PathLike[bytes]], int]):
    with open(file_name, "rb") as audio:
        source = {"buffer": audio, "mimetype": "audio/wav"}
        response = await deepgram.transcription.prerecorded(source, {'language': 'id'})
        return response["results"]["channels"][0]["alternatives"][0]["transcript"]

def log(log: str):
    # Cetak dan tulis ke status.txt
    print(log)
    with open("status.txt", "w") as f:
        f.write(log)

def introduction():
    # Pengenalan
    intro_text = """
    Halo! Saya Nathan, teman diskusi kamu hari ini. Saya di sini untuk membantu kamu belajar tentang sejarah Indonesia. 
    Kamu mau belajar tentang apa hari ini?
    """
    
    audio = elevenlabs.generate(text=intro_text, voice="d888tBvGmQT2u05J1xTv", model="eleven_multilingual_v2")

    elevenlabs.save(audio, "audio/intro.wav")
    
    # Putar suara pengenalan
    mixer.Sound("audio/intro.wav").play()
    pygame.time.wait(int(mixer.Sound("audio/intro.wav").get_length() * 1000))
    
    print(intro_text)

if __name__ == "__main__":
    introduction()
    while True:
        # Rekam audio
        speech_to_text()
        log("Selesai mendengarkan")

        # Transkripsi audio
        current_time = time()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        transcript = loop.run_until_complete(transcribe(RECORDING_PATH))
        with open("conv.txt", "a") as f:
            f.write(f"{transcript}\n")
        transcription_time = time() - current_time
        log(f"Transkripsi selesai dalam {transcription_time:.2f} detik.")

        # Dapatkan respon dari GPT
        current_time = time()
        context += f"\nPengguna: {transcript}\nNathan: "
        response = request_gpt(context)
        context += response
        gpt_time = time() - current_time
        log(f"Respon GPT selesai dalam {gpt_time:.2f} detik.")

        # Konversi respon menjadi audio dengan ElevenLabs
        current_time = time()
        audio = elevenlabs.generate(
            text=response, voice="d888tBvGmQT2u05J1xTv", model="eleven_multilingual_v2"
        )
        elevenlabs.save(audio, "audio/response.wav")
        audio_time = time() - current_time
        log(f"Audio selesai dibuat dalam {audio_time:.2f} detik.")

        # Putar respon
        log("Memutar respon...")
        sound = mixer.Sound("audio/response.wav")
        # Tambahkan respon sebagai baris baru di conv.txt
        with open("conv.txt", "a") as f:
            f.write(f"{response}\n")
        sound.play()
        pygame.time.wait(int(sound.get_length() * 1000))
        print(f"\n --- User: {transcript}\n --- AI: {response}\n")
