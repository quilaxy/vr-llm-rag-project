# Gunakan base image Python
FROM python:3.10-slim

# Set lingkungan kerja
WORKDIR /app

# Salin file requirements.txt ke dalam container
COPY requirements.txt requirements.txt

# Instal dependensi sistem yang dibutuhkan
RUN apt-get update && apt-get install -y \
    gcc \
    libasound2-dev \
    libportaudio2 \
    libportaudiocpp0 \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# Instal dependensi Python
RUN pip install --no-cache-dir -r requirements.txt

# Salin semua file proyek ke dalam container
COPY . .

# Salin file service.json ke dalam container
COPY service.json /app/service.json

# Set variabel lingkungan untuk Google Cloud
ENV GOOGLE_APPLICATION_CREDENTIALS="/app/service.json"

# Jalankan aplikasi Fast API
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]

# Flask
CMD ["gunicorn", "-b", "0.0.0.0:8080", "main:app"]
