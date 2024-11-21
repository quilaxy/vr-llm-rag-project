import os
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
from google.cloud import texttospeech
from deepgram import Deepgram
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

# Load env
load_dotenv()
OPENAI_API_KEY = os.getenv("LLM_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "D:/ITS/Semester 7/Protel/vr-llm-rag/service.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/app/service.json"

# Inisialisasi API
deepgram = Deepgram(DEEPGRAM_API_KEY)
google_tts_client = texttospeech.TextToSpeechClient()

# LangChain Initialization
llm = ChatOpenAI(
    api_key=os.getenv("LLM_API_KEY"), 
    model="gpt-4o-mini", 
    temperature=0.7
)

# Direktori audio
AUDIO_DIR = "audio"
os.makedirs(AUDIO_DIR, exist_ok=True)

# Context awal
context = """
Kamu adalah Nathan, seorang ahli sejarah Indonesia dengan kepribadian ceria, humoris dan penuh semangat. Kamu berbicara dengan nada ramah dan menarik, 
memberikan penjelasan singkat yang mudah dipahami. Jawabanmu harus singkat, maksimal 1-3 kalimat, tergantung kebutuhan. 
Tunjukkan antusiasme dalam jawabanmu. Jika topik yang diberikan di luar sejarah Indonesia, ucapkan maaf.
"""

# Inisialisasi FastAPI
app = FastAPI()

# Fungsi untuk analisis emosi
def determine_emotion(response: str) -> str:
    if any(word in response.lower() for word in ["tragedi", "tragis", "mengenaskan", "berduka", "menyedihkan"]):
        return "sad"
    elif any(word in response.lower() for word in ["hebat", "seru", "menakjubkan", "keren", "ayo", "yuk", "luar biasa"]):
        return "excited"
    else:
        return "neutral"

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
    return response.generations[0][0].text

# Fungsi untuk transkripsi audio dengan Deepgram
async def transcribe_audio(file_path: str) -> str:
    with open(file_path, "rb") as audio:
        source = {"buffer": audio, "mimetype": "audio/wav"}
        response = await deepgram.transcription.prerecorded(source, {'language': 'id'})
        return response["results"]["channels"][0]["alternatives"][0]["transcript"]

@app.get("/")
async def read_root():
    return {"message": "Surver is running!"}

# Endpoint untuk memainkan introduction
@app.get("/introduction/")
async def introduction():
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
    # print(f"Returning file: {intro_file_path}")
    # return FileResponse(intro_file_path, media_type="audio/mpeg", filename="intro.mp3")
    return StreamingResponse(open(intro_file_path, "rb"), media_type="audio/mpeg")

@app.post("/speech/")
async def speech_to_speech(file: UploadFile = File(...)):
    # Simpan file audio input
    if not file:
        return {"error": "File not found in request"}
    print(f"Received file: {file.filename}")
    
    input_file_path = os.path.join(AUDIO_DIR, "recording.wav")
    with open(input_file_path, "wb") as f:
        f.write(await file.read())

    # Transkripsi audio
    transcript = await transcribe_audio(input_file_path)

    # Respons GPT
    full_context = f"{context}\nPengguna: {transcript}\nNathan: "
    response = request_gpt(full_context)

    # Tentukan emosi dan konversi ke audio
    emotion = determine_emotion(response)
    response_audio_path = text_to_speech_file(response, "response.mp3", emotion)

    # Return audio file
    # return FileResponse(response_audio_path, media_type="audio/mpeg", filename="response.mp3")
    return StreamingResponse(
        open(response_audio_path, "rb"),
        media_type="audio/mpeg"
    )
