#!/usr/bin/env bash
set -euxo pipefail

# Устанавливаем Python-зависимости
pip install --upgrade pip
pip install -r requirements.txt

# Ставим ffmpeg (статические бинарники под Linux x64)
# Источник: John Van Sickle — стабильные статические сборки
FFMPEG_URL="https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz"
curl -L "$FFMPEG_URL" -o ffmpeg.tar.xz
mkdir -p .ffmpeg
tar -xJf ffmpeg.tar.xz --strip-components=1 -C .ffmpeg
mkdir -p bin
cp .ffmpeg/ffmpeg .ffmpeg/ffprobe bin/
chmod +x bin/ffmpeg bin/ffprobe
echo "ffmpeg version: $(bin/ffmpeg -version | head -n1)"
echo "ffprobe version: $(bin/ffprobe -version | head -n1)"
