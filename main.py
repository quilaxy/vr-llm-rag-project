from rag import RAGManager
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import os
from record import speech_to_text
import asyncio
from deepgram import Deepgram
from time import time
# import pygame
# from pygame import mixer
# import elevenlabs

# Load API keys
from dotenv import load_dotenv
load_dotenv()

# Inisiasi API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
# elevenlabs.set_api_key(os.getenv("ELEVENLABS_API_KEY"))

# Initialize APIs
rag_manager = RAGManager(embedding_model_api_key=os.getenv("EMBED_API_KEY"))
deepgram = Deepgram(DEEPGRAM_API_KEY)
# mixer.init()

# File untuk menyimpan rekaman audio
RECORDING_PATH = "audio/recording.wav"

# Fungsi untuk transkripsi audio menggunakan Deepgram
async def transcribe_audio(file_path):
    with open(file_path, "rb") as audio:
        source = {"buffer": audio, "mimetype": "audio/wav"}
        try:
            response = await deepgram.transcription.prerecorded(source, {'language': 'id' })
            # print(f"Deepgram Response: {response}")  # Cetak keseluruhan respons
            transcript = response["results"]["channels"][0]["alternatives"][0]["transcript"]
            return transcript
        except Exception as e:
            print(f"Error transcribing audio: {e}")
            return ""

# Fungsi untuk menawarkan materi
def offer_material_choice():
    return """
    Halo! Selamat datang di kelas sejarah! Apa yang mau kamu pelajari hari ini? Di kelas ini kami menyediakan beberapa topik, seperti:
    - Peristiwa Rengasdengklok dan Proklamasi
    - Peristiwa 10 Nopember
    - Konferensi Meja Bundar
    """

# Fungsi untuk memproses pilihan materi berdasarkan input pengguna
def get_material_choice(user_input):
    user_input = user_input.lower()

    choices = {
        "Peristiwa Rengasdengklok": ["peristiwa rengasdengklok", "rengasdengklok", "proklamasi", "proklamasi kemerdekaan"],
        "Peristiwa 10 Nopember": ["peristiwa 10 nopember", "10 nopember", "10 november"],
        "Konferensi Meja Bundar": ["konferensi meja bundar", "meja bundar", "kmb"]
    }

    for material, keywords in choices.items():
        if any(keyword in user_input for keyword in keywords):
            return material
    
    return None

# Fungsi untuk membuat ringkasan/overview materi
def generate_ai_overview(llm, material_choice, retrieved_docs):
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    summary_prompt = [
        SystemMessage(content="Kamu adalah asisten yang pandai merangkum informasi."),
        HumanMessage(content=f"Tolong buatkan ringkasan atau penjelasan singkat tentang topik ini dalam satu paragraf: {material_choice}.\n\n{context}")
    ]

    summary_response = llm.invoke(summary_prompt)

    return summary_response.content

# Main interaction function
def interact_with_user():
    print(offer_material_choice())

    # Merekam audio dari pengguna
    print("Silakan sebutkan topik yang ingin kamu pelajari.")
    speech_to_text()  # Rekam suara dan simpan ke file WAV

    # Transkripsi suara yang direkam
    file_path = RECORDING_PATH
    loop = asyncio.get_event_loop()
    user_input = loop.run_until_complete(transcribe_audio(file_path)).strip()
    print(f"Hasil Transkripsi: {user_input}")

    # Proses pilihan materi berdasarkan hasil transkripsi
    material_choice = get_material_choice(user_input)

    if material_choice:
        print(f"\n\nKamu memilih untuk mempelajari tentang {material_choice}.")

        # Mengambil dokumen terkait
        retrieved_docs = rag_manager.query_faiss(material_choice, "overview")

        # Inisiasi LLM
        llm = ChatOpenAI(api_key=os.getenv("LLM_API_KEY"), model="gpt-4o-mini", temperature=0.7)

        # Membuat overview materi
        overview = generate_ai_overview(llm, material_choice, retrieved_docs)
        print(f"Overview dari materi:\n{overview}")

        while True:
            # Rekam pertanyaan pengguna dan transkrip
            print(f"Silakan tanyakan sesuatu tentang {material_choice}...")
            speech_to_text()
            query = loop.run_until_complete(transcribe_audio(file_path)).strip()

            if not query:
                print("Terima kasih! Sampai jumpa lagi.")
                break

            # Mengambil dokumen terkait pertanyaan
            retrieved_docs = rag_manager.query_faiss(material_choice, query)
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])

            # Menyiapkan pesan untuk LLM
            messages = [
                SystemMessage(content="Kamu seorang guru sejarah. Jawablah pertanyaan yang diberikan sesuai konteks yang diberikan. Jawablah seringkas mungkin sesuai pertanyaan yang diberikan. Susun jawaban dalam kalimat. Jika tidak ada data yang berkaitan dengan konteks di bawah, jawab dengan 'Saya tidak tahu'."),
                HumanMessage(content=f"Pertanyaan: {query}\n\nKonteks:\n{context}")
            ]

            # Dapatkan respons dari LLM
            response = llm.invoke(messages)
            print(f"AI's Response: {response.content}")

    else:
        print("Pilihan tidak valid. Silakan coba lagi.")

# Jalankan alur interaksi
if __name__ == "__main__":
    interact_with_user()
