# ------- Test record ------

# from record import speech_to_text
# speech_to_text()


# -------- Test api key deepgram --------

# import os
# DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
# print(f"Deepgram API Key: {DEEPGRAM_API_KEY}")




# ------- test deepgram manual dari file audio ----------

import os
import asyncio
from deepgram import Deepgram

# Load API key dari environment
from dotenv import load_dotenv
load_dotenv()

# Ambil API key Deepgram dari environment
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
if not DEEPGRAM_API_KEY:
    raise ValueError("API Key Deepgram tidak ditemukan.")

# Inisiasi Deepgram client
deepgram = Deepgram(DEEPGRAM_API_KEY)

# Path ke file audio (pastikan file ini sudah ada)
RECORDING_PATH = "audio/recording.wav"

# Fungsi untuk mengirim file WAV ke Deepgram dan menampilkan respons lengkap
async def test_deepgram(file_path):
    try:
        # Buka file audio yang akan ditranskripsi
        with open(file_path, "rb") as audio:
            source = {"buffer": audio, "mimetype": "audio/wav"}

            # Panggil API Deepgram untuk melakukan transkripsi
            response = await deepgram.transcription.prerecorded(source, {'language': 'id' })

            # Cetak respons lengkap dari Deepgram untuk debugging
            print(f"Deepgram Full Response: {response}")

            # Ambil transkrip dari hasil Deepgram
            transcript = response["results"]["channels"][0]["alternatives"][0]["transcript"]
            print(f"Hasil Transkripsi: {transcript}")

    except Exception as e:
        print(f"Terjadi kesalahan saat mengirim file ke Deepgram: {e}")

# Main program untuk menjalankan fungsi di atas
if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(test_deepgram(RECORDING_PATH))

