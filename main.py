import os
from flask import Flask, request, Response, jsonify, send_file
from flask_cors import CORS
from dotenv import load_dotenv
from google.cloud import texttospeech
from deepgram import Deepgram
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
import asyncio
import re

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("LLM_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "D:/ITS/Semester 7/Protel/vr-llm-rag/service.json"   #lokal
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/app/service.json"  #docker

# Inisialisasi API
deepgram = Deepgram(DEEPGRAM_API_KEY)
google_tts_client = texttospeech.TextToSpeechClient()

# LangChain Initialization
llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model="gpt-4o-mini",
    temperature=0.7,
)

# Direktori audio
AUDIO_DIR = "audio"
HISTORY_FILE = "conversation_history.txt"
os.makedirs(AUDIO_DIR, exist_ok=True)

# Context awal
context = """
Kamu adalah Nathan, seorang ahli sejarah Indonesia dengan kepribadian ceria, humoris, dan penuh semangat. 
Kamu merupakan seorang yang asik diajak berdiskusi. 
Kamu berbicara dengan nada ramah dan menarik, memberikan penjelasan singkat yang mudah dipahami. 
Kamu harus selalu menggunakan kata ganti orang pertama ketika berbicara tentang dirimu.
Jawabanmu harus sangat singkat, maksimal 1-4 kalimat, tergantung kebutuhan. Hindari penjelasan yang terlalu panjang atau bertele-tele. 
Tunjukkan antusiasme dalam jawabanmu. Kamu HANYA membahas topik yang berkaitan dengan sejarah Indonesia.
Jika topik yang diberikan di luar sejarah Indonesia, ucapkan permintaan maaf.
"""

# Inisialisasi Flask
app = Flask(__name__)
CORS(app)

# Fungsi untuk menghapus riwayat percakapan
def clear_history():
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        f.write("")

# Fungsi untuk menambah riwayat percakapan
def append_to_history(entry: str):
    with open(HISTORY_FILE, "a", encoding="utf-8") as f:
        f.write(entry + "\n")

# Fungsi untuk membaca riwayat percakapan
def read_history() -> str:
    if not os.path.exists(HISTORY_FILE):
        return ""
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        return f.read()

# Fungsi untuk analisis emosi
def determine_emotion(response: str) -> str:
    if any(word in response.lower() for word in ["tragedi", "tragis", "mengenaskan", "berduka", "menyedihkan"]):
        return "sad"
    elif any(word in response.lower() for word in ["saya", "hebat", "seru", "menakjubkan", "keren", "ayo", "yuk", "luar biasa"]):
        return "excited"
    else:
        return "neutral"

# Fungsi untuk menghapus simbol yang tidak perlu  
def sanitize_text(text: str) -> str:
    cleaned_text = re.sub(r'[^\w\s.,?!\-%=]', '', text)  # Hanya simpan huruf, angka, spasi, tanda baca umum
    return cleaned_text.strip()

# Fungsi untuk memproses teks ke audio
def text_to_speech_file(text: str, filename: str, emotion: str) -> str:
    ssml_text = f"""
    <speak>
        {(text.lower())
            .replace(", kan", "<emphasis level='strong'><prosody pitch='+6st' rate='1.2'>kan?</prosody><break time='100ms'/>")
            .replace(", lho", "<prosody pitch='+3st' rate='0.8'>lohh</prosody></emphasis><break time='100ms'/>")
            .replace(", bukan", "<prosody pitch='+4st' rate='1.2'>bukan?</prosody><break time='100ms'/>")
            .replace(", yuk", "<prosody pitch='+6st' rate='1.2'>yuk!</prosody><break time='100ms'/>")
            .replace(", ya", "<prosody pitch='+6st' rate='1.2'>ya!</prosody><break time='100ms'/>")
            .replace("tentu!", "<prosody pitch='+2st' rate='1.2'>tentu</prosody><break time='400ms'/>")
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
        pitch = 2.0 if emotion == "excited" else 1.0 if emotion == "sad" else 1.8,
        speaking_rate = 1.2 if emotion == "excited" else 1.0 if emotion == "sad" else 1.1
    )

    response = google_tts_client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    save_file_path = os.path.join(AUDIO_DIR, filename)
    with open(save_file_path, "wb") as out:
        out.write(response.audio_content)
    
    return save_file_path

# Fungsi untuk meminta respons dari GPT
def request_gpt(prompt: str) -> str:
    messages = [HumanMessage(content=prompt)]
    response = llm.generate(messages=[messages])
    return response.generations[0][0].text.strip()

# Fungsi untuk transkripsi audio dengan Deepgram
async def transcribe_audio(file_path: str) -> str:
    with open(file_path, "rb") as audio:
        source = {"buffer": audio, "mimetype": "audio/wav"}
        response = await deepgram.transcription.prerecorded(source, {'language': 'id'})
        return response["results"]["channels"][0]["alternatives"][0]["transcript"]

@app.route("/")
def home():
    return jsonify({"message": "Server is running!"})

# Get untuk cek suara
@app.route("/intro/", methods=["GET"]) 
def introduction():
    intro_text = """
    <speak>
        <prosody rate="1.2" pitch="+2st">
            HALOO!!!
        </prosody>
        <break time="200ms"/> 
        <prosody rate="1.2" pitch="+1.5st">
            Saya Nathan, teman diskusi kamu hari ini. Saya di sini untuk membantu kamu belajar tentang sejarah Indonesia. 
            Kamu mau belajar tentang apa hari ini?
        </prosody>
    </speak>
    """
    intro_file_path = text_to_speech_file(intro_text, "intro.mp3", "excited")
    
    return send_file(intro_file_path, as_attachment=True)

@app.route("/speech/", methods=["POST"])
def speech_to_speech():
    try:
        file = request.files.get("file")
        if not file:
            return jsonify({"error": "File not found in request"}), 400

        input_file_path = os.path.join(AUDIO_DIR, "recording.wav")
        file.save(input_file_path)

        # Transkripsi audio
        transcript = asyncio.run(transcribe_audio(input_file_path))
        # print(f"Transcript: {transcript}")

        # Respons GPT
        conversation_history = read_history()
        full_context = f"{context}\n{conversation_history}\nPengguna: {transcript}\nNathan: "
        response = request_gpt(full_context)
        # print(f"GPT Response: {response}")

        sanitized_response = sanitize_text(response)

        # Simpan riwayat percakapan
        append_to_history(f"Pengguna: {transcript}")
        append_to_history(f"Nathan: {sanitized_response}")

        # Tentukan emosi dan konversi ke audio
        emotion = determine_emotion(sanitized_response)
        response_audio_path = text_to_speech_file(sanitized_response, "response.mp3", emotion)

        return send_file(
            response_audio_path,
            as_attachment=True,
            download_name="response.mp3",
            mimetype="audio/mpeg"
        )

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "Internal Server Error"}), 500

@app.route("/clear/", methods=["POST"])
def clear_history_route():
    try:
        clear_history()
        return jsonify({"message": "Riwayat percakapan berhasil dihapus"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
