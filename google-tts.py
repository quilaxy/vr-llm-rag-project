import os
from os import PathLike
from time import time
import asyncio
from typing import Union

from dotenv import load_dotenv
from google.cloud import texttospeech
from deepgram import Deepgram
import pygame
from pygame import mixer
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from record import speech_to_text


# Load env
load_dotenv()
OPENAI_API_KEY = os.getenv("LLM_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "D:/ITS/Semester 7/Protel/vr-llm-rag/service.json"

# Inisialisasi API
deepgram = Deepgram(DEEPGRAM_API_KEY)
google_tts_client = texttospeech.TextToSpeechClient()
mixer.init()

# Inisialisasi OpenAI LangChain
llm = ChatOpenAI(
    api_key=os.getenv("LLM_API_KEY"), 
    model="gpt-4o-mini", 
    temperature=0.7
)

context = """
Kamu adalah Nathan, seorang ahli sejarah Indonesia dengan kepribadian ceria, humoris dan penuh semangat. Kamu berbicara dengan nada ramah dan menarik, 
memberikan penjelasan singkat yang mudah dipahami. Jawabanmu harus singkat, maksimal 1-3 kalimat, tergantung kebutuhan. 
Tunjukkan antusiasme dalam jawabanmu. Jika topik yang diberikan di luar sejarah Indonesia, ucapkan maaf.
"""

RECORDING_PATH = "audio/recording.wav"

conversation_history = []

def update_conversation_history(user_input: str, ai_response: str):
    conversation_history.append(f"Pengguna: {user_input}\nNathan: {ai_response}")
    if len(conversation_history) > 6:
        conversation_history.pop(0)

def get_context_with_history():
    history_text = "\n".join(conversation_history)
    return f"{context}\n{history_text}\nPengguna:"

def clear_history():
    """Menghapus atau mereset file history percakapan."""
    with open("conversation_history.txt", "w", encoding="utf-8") as f:
        f.write("")

# Fungsi untuk analisis sentimen
def determine_emotion(response: str) -> str:
    if any(word in response.lower() for word in ["tragedi", "tragis", "mengenaskan", "berduka", "peristiwa menyedihkan", "jatuhnya korban", "pemberontakan", "pertempuran berdarah"]):
        return "sad"
    elif any(word in response.lower() for word in ["hebat", "seru", "menakjubkan", "keren", "ayo", "yuk", "luar biasa", "jelajahi", "lihat", "tentu!", "sama-sama!", "mantap", "senang"]):
        return "excited"
    else:
        return "neutral"

# Konversi teks menjadi audio WAV menggunakan Google Text-to-Speech
def text_to_speech_file(text: str, filename: str, emotion: str) -> str:
    ssml_text = f"""
    <speak>
        {(text.lower())
            .replace(", kan", "<emphasis level='strong'><prosody pitch='+6st' rate='1.2'>kan?</prosody><break time='100ms'/>")
            .replace(", lho", "<prosody pitch='+3st' rate='0.8'>lohh</prosody></emphasis><break time='100ms'/>")
            .replace(", bukan", "<prosody pitch='+4st' rate='1.2'>bukan?</prosody><break time='100ms'/>")
            .replace(", yuk", "<prosody pitch='+6st' rate='1.2'>yuk!</prosody><break time='100ms'/>")
            .replace(", ya", "<prosody pitch='+6st' rate='1.2'>ya!</prosody><break time='100ms'/>")
            .replace("yuk, ", "<prosody pitch='+6st' rate='1.2'>yuk</prosody><break time='100ms'/>")
            .replace("tentu!", "<prosody pitch='+2st' rate='1.2'>tentu</prosody><break time='100ms'/>")
        }
    </speak>
    """

    synthesis_input = texttospeech.SynthesisInput(ssml=ssml_text)

    voice = texttospeech.VoiceSelectionParams(
        language_code="id-ID",
        ssml_gender=texttospeech.SsmlVoiceGender.MALE
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3,
        pitch = 2.0 if emotion == "excited" else 0 if emotion == "sad" else 1.5,
        speaking_rate = 1.2 if emotion == "excited" else 1.0 if emotion == "sad" else 1.1
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
    return response.generations[0][0].text.strip()

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
    <speak>
        <prosody rate="1.2" pitch="+2st">
            HALOO!!!
        </prosody>
        <break time="200ms"/> 
        <prosody rate="1.2" pitch="+1.5st">
            Saya Nathan, <break time="300ms"/> teman diskusi kamu hari ini. Saya di sini untuk membantu kamu belajar tentang sejarah Indonesia. 
            Kamu mau belajar tentang apa hari ini?
        </prosody>
    </speak>
    """

    intro_file_path = text_to_speech_file(intro_text, "intro.wav", "excited")
    
    mixer.Sound(intro_file_path).play()
    pygame.time.wait(int(mixer.Sound(intro_file_path).get_length() * 1000))
    print(intro_text)

# if __name__ == "__main__":
#     introduction()
#     while True:
#         # Rekam audio
#         log("Mendengarkan...")
#         speech_to_text()
#         log("Selesai mendengarkan")

#         # Transkripsi audio
#         current_time = time()
#         loop = asyncio.new_event_loop()
#         asyncio.set_event_loop(loop)
#         transcript = loop.run_until_complete(transcribe(RECORDING_PATH))
#         transcription_time = time() - current_time
#         log(f"Transkripsi selesai dalam {transcription_time:.2f} detik.")
#         print(f"Transkripsi: {transcript}")

#         # Dapatkan respon dari GPT
#         context += f"\nPengguna: {transcript}\nNathan: "
#         response = request_gpt(context)
#         context += response

#         emotion = determine_emotion(response)

#         # Konversi respons menjadi audio
#         response_audio_file = text_to_speech_file(response, "response.mp3", emotion)
#         sound = mixer.Sound(response_audio_file)
#         with open("conv.txt", "w", encoding="utf-8") as f:
#             f.write(f"{response}\n")
#         sound.play()
#         pygame.time.wait(int(sound.get_length() * 1000))
        
#         # print(f"\nUser: {transcript}\nAI: {response}")
#         print(f"\nAI: {response}")


if __name__ == "__main__":
    clear_history()
    # introduction()
    while True:
        log("Mendengarkan...")
        speech_to_text()
        log("Selesai mendengarkan")

        # Transkripsi
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        transcript = loop.run_until_complete(transcribe(RECORDING_PATH))
        print(f"Transkripsi: {transcript}")

        # Susun konteks dengan histori
        full_context = get_context_with_history() + f" {transcript}\nNathan:"

        # Minta respons dari GPT
        response = request_gpt(full_context)

        # Perbarui histori percakapan
        update_conversation_history(transcript, response)

        # Simpan histori ke file
        with open("conversation_history.txt", "w", encoding="utf-8") as f:
            f.write("\n\n".join(conversation_history))

        # Tentukan emosi
        emotion = determine_emotion(response)

        # Konversi ke audio
        response_audio_file = text_to_speech_file(response, "response.mp3", emotion)
        sound = mixer.Sound(response_audio_file)
        sound.play()
        pygame.time.wait(int(sound.get_length() * 1000))

        # Debugging
        print(f"\nUser: {transcript}")
        print(f"AI: {response}")
        print(f"Emotion: {emotion}")
        print(f"Conversation History: {conversation_history}")
