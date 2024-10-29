import io
import time
import wave
import pyaudio
from pathlib import Path
from rhasspysilence import WebRtcVadRecorder, VoiceCommand, VoiceCommandResult

# Initialize PyAudio for recording
pa = pyaudio.PyAudio()

def speech_to_text() -> None:
    """
    Merekam audio sampai terdeteksi tidak ada suara dan menyimpannya dalam file WAV.
    """
    recorder = WebRtcVadRecorder(
        vad_mode=3,  # Mode deteksi suara, 3 = paling sensitif
        silence_seconds=4,  # Hentikan rekaman setelah 4 detik keheningan
    )
    recorder.start()

    # Direktori dan nama file untuk menyimpan rekaman
    wav_sink = "audio/"
    wav_filename = "recording"

    # Cek apakah direktori sudah ada, jika tidak buat baru
    wav_sink_path = Path(wav_sink)
    if not wav_sink_path.exists():
        wav_sink_path.mkdir(parents=True, exist_ok=True)

    # Jika direktori valid
    wav_dir = wav_sink_path if wav_sink_path.is_dir() else None

    audio_source = pa.open(
        rate=16000,
        format=pyaudio.paInt16,
        channels=1,
        input=True,
        frames_per_buffer=960,
    )
    audio_source.start_stream()

    def buffer_to_wav(buffer: bytes) -> bytes:
        """
        Mengonversi buffer audio mentah ke format WAV.
        """
        rate = int(16000)
        width = int(2)
        channels = int(1)

        with io.BytesIO() as wav_buffer:
            wav_file: wave.Wave_write = wave.open(wav_buffer, mode="wb")
            with wav_file:
                wav_file.setframerate(rate)
                wav_file.setsampwidth(width)
                wav_file.setnchannels(channels)
                wav_file.writeframesraw(buffer)

            return wav_buffer.getvalue()

    try:
        print("Mendengarkan...")

        chunk = audio_source.read(960)
        while chunk:
            # Proses chunk untuk mendeteksi suara/keheningan
            voice_command = recorder.process_chunk(chunk)

            if voice_command and voice_command.result != VoiceCommandResult.FAILURE:
                # Suara selesai, simpan audio
                audio_data = recorder.stop()
                if wav_dir:
                    # Menyimpan rekaman dalam direktori
                    wav_path = (wav_dir / time.strftime(wav_filename)).with_suffix(".wav")
                    wav_bytes = buffer_to_wav(audio_data)
                    wav_path.write_bytes(wav_bytes)
                    print(f"Rekaman disimpan di: {wav_path}")
                    break
            # Membaca chunk berikutnya
            chunk = audio_source.read(960)

    finally:
        try:
            audio_source.stop_stream()
            audio_source.close()
        except Exception as e:
            print(f"Error saat menutup stream: {e}")

if __name__ == "__main__":
    speech_to_text()
