#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import csv
import cv2
import numpy as np
import subprocess
import sys
from datetime import timedelta
from scipy.fft import rfft, rfftfreq
from scipy.io import wavfile

# tqdm: если нет в системе — работаем без прогресса
try:
    from tqdm import tqdm
except Exception:  # ImportError
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else range(0)

# чтобы вывод появлялся сразу (без буферизации)
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

# =======================
#        SETTINGS
# =======================
VIDEO_FOLDER = "./videos"

OUTPUT_CSV_EVENTS = "report_black_glitch_tone2.csv"   # подробный CSV по событиям (как раньше)
OUTPUT_CSV_SUMMARY = "report_summary.csv"             # сводка по каждому видео

TEMP_AUDIO = "temp_audio.wav"

# Пороги/параметры
MIN_GLITCH_DURATION = 10          # сек (для GLITCH и RUIS/STRIPES)
BLACKDETECT_MIN_DURATION = 10     # сек
FREEZE_MIN_DURATION = 5           # сек
TONE_MIN_DURATION = 5             # сек

# blackdetect — чувствительнее к почти-чёрному
BLACKDETECT_PIX_TH = 0.10         # пиксель «чёрный», если Y < 10% (~25/255)
BLACKDETECT_PIC_TH = 0.98

# 1 kHz
TONE_HZ = 1000
HZ_TOLERANCE = 50

# RUIS/STRIPES (серый + шум/полосы)
RUIS_SAT_MAX = 40.0               # средняя насыщенность (S) в HSV <= 40 → «серый»
RUIS_LAP_VAR_MIN = 120.0          # «шумность» кадра через Лапласиан
RUIS_STRIPE_STD_MIN = 10.0        # «полосатость» (std средних по столбцам)
RUIS_FPS_SAMPLE = 1               # брать ~1 кадр/сек для скорости

# Какие расширения считаем видео
VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".avi", ".m4v")


# =======================
#       HELPERS
# =======================
def to_hms(seconds: float) -> str:
    return str(timedelta(seconds=int(round(seconds))))

def seconds_to_mmss(seconds: float) -> str:
    seconds = int(round(seconds))
    m, s = divmod(seconds, 60)
    return f"{m:02d}:{s:02d}"

def hms_to_seconds(hms: str) -> float:
    """'HH:MM:SS' → seconds (robust for 'MM:SS' too)."""
    parts = [float(p) for p in hms.split(":")]
    if len(parts) == 3:
        h, m, s = parts
        return h*3600 + m*60 + s
    elif len(parts) == 2:
        m, s = parts
        return m*60 + s
    else:
        return float(parts[0])

def natural_sort_key(s: str):
    # натуральная сортировка: file2 < file10
    return [int(t) if t.isdigit() else t.lower() for t in re.findall(r'\d+|\D+', s)]

def get_video_duration_seconds(filepath: str) -> float:
    """Сначала ffprobe, если нет — OpenCV."""
    try:
        cmd = [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=nw=1:nk=1", filepath
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()
        dur = float(out)
        if dur > 0:
            return dur
    except Exception:
        pass

    cap = cv2.VideoCapture(filepath)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    cap.release()
    return float(frames / fps) if fps > 0 else 0.0

def merge_intervals(intervals):
    """Объединяет пересекающиеся интервалы [(start_sec, end_sec), ...]."""
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = [list(intervals[0])]
    for start, end in intervals[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:  # пересекаются или касаются
            merged[-1][1] = max(last_end, end)
        else:
            merged.append([start, end])
    return [(a, b) for a, b in merged]


# =======================
#     DETECTORS
# =======================
def detect_black_segments(filepath):
    print("   ⏳ blackdetect…", flush=True)
    results = []
    ffmpeg_cmd = [
        "ffmpeg", "-hide_banner", "-i", filepath,
        "-vf", f"blackdetect=d={BLACKDETECT_MIN_DURATION}:pix_th={BLACKDETECT_PIX_TH}:pic_th={BLACKDETECT_PIC_TH}",
        "-an", "-f", "null", "-"
    ]
    result = subprocess.run(ffmpeg_cmd, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True)
    for line in result.stderr.splitlines():
        try:
            match = re.search(r'black_start:(\d+\.?\d*)\s+black_end:(\d+\.?\d*)\s+black_duration:(\d+\.?\d*)', line)
            if match:
                start = float(match.group(1))
                end = float(match.group(2))
                duration = float(match.group(3))
                results.append({
                    "type": "BLACK",
                    "start": to_hms(start),
                    "end": to_hms(end),
                    "duration": duration,
                    "details": "black screen"
                })
        except Exception:
            continue
    print("   ✅ blackdetect done", flush=True)
    return results


def detect_glitches(filepath, crop_top_ratio=0.0):
    """Простые цветовые аномалии: зелёный/розовый/пересвет."""
    results = []
    cap = cv2.VideoCapture(filepath)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fps <= 0 or frame_count <= 0:
        cap.release()
        return results

    in_glitch = False
    glitch_start = None

    for i in tqdm(range(frame_count),
                  desc=f"   🎛 GLITCH {os.path.basename(filepath)}",
                  unit="f",
                  leave=False):
        ret, frame = cap.read()
        if not ret:
            break

        if crop_top_ratio > 0.0:
            h = frame.shape[0]
            cut = int(h * crop_top_ratio)
            if cut < h:
                frame = frame[cut:, :]

        avg_color = frame.mean(axis=0).mean(axis=0)
        r, g, b = float(avg_color[2]), float(avg_color[1]), float(avg_color[0])

        green_glitch = g > 180 and g > r and g > b
        pink_glitch = r > 180 and b > 180 and g < 130
        oversaturated = (r > 230 or g > 230 or b > 230)

        is_glitch = green_glitch or pink_glitch or oversaturated

        if is_glitch and not in_glitch:
            in_glitch = True
            glitch_start = i
        elif not is_glitch and in_glitch:
            glitch_end = i
            glitch_duration = (glitch_end - glitch_start) / fps
            if glitch_duration >= MIN_GLITCH_DURATION:
                results.append({
                    "type": "GLITCH",
                    "start": to_hms(glitch_start / fps),
                    "end": to_hms(glitch_end / fps),
                    "duration": glitch_duration,
                    "details": "green/pink/oversaturated anomaly"
                })
            in_glitch = False

    if in_glitch:
        glitch_end = frame_count
        glitch_duration = (glitch_end - glitch_start) / fps
        if glitch_duration >= MIN_GLITCH_DURATION:
            results.append({
                "type": "GLITCH",
                "start": to_hms(glitch_start / fps),
                "end": to_hms(glitch_end / fps),
                "duration": glitch_duration,
                "details": "green/pink/oversaturated anomaly (end)"
            })

    cap.release()
    return results


def detect_freezes(filepath):
    print("   ⏳ freezedetect…", flush=True)
    results = []
    ffmpeg_cmd = [
        "ffmpeg", "-hide_banner", "-i", filepath,
        "-vf", f"freezedetect=n=0.003:d={FREEZE_MIN_DURATION}",
        "-an", "-f", "null", "-"
    ]
    result = subprocess.run(ffmpeg_cmd, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True)
    freeze_start = None

    for line in result.stderr.splitlines():
        line = line.strip()
        if "freeze_start:" in line:
            try:
                freeze_start = float(line.split("freeze_start:")[1])
            except Exception:
                freeze_start = None
        elif "freeze_end:" in line and freeze_start is not None:
            try:
                freeze_end = float(line.split("freeze_end:")[1])
                duration = freeze_end - freeze_start
                results.append({
                    "type": "FREEZE",
                    "start": to_hms(freeze_start),
                    "end": to_hms(freeze_end),
                    "duration": duration,
                    "details": "frozen frame"
                })
            except Exception:
                pass
            freeze_start = None

    print("   ✅ freezedetect done", flush=True)
    return results


def detect_1khz_tone(filepath):
    print("   ⏳ 1kHz tone detect…", flush=True)
    results = []
    extract_cmd = [
        "ffmpeg", "-y", "-i", filepath, "-vn", "-ac", "1", "-ar", "44100", "-f", "wav", TEMP_AUDIO
    ]
    subprocess.run(extract_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if not os.path.exists(TEMP_AUDIO):
        print("   ⚠️ no audio extracted", flush=True)
        return results

    samplerate, data = wavfile.read(TEMP_AUDIO)
    if data.ndim > 1:
        data = data[:, 0]

    window_size = int(samplerate * TONE_MIN_DURATION)
    step = int(window_size / 2)

    for start in tqdm(range(0, len(data) - window_size, step),
                      desc="   🎚 audio windows",
                      unit="win",
                      leave=False):
        window = data[start:start+window_size]
        yf = np.abs(rfft(window))
        xf = rfftfreq(len(window), 1 / samplerate)
        idx = int(np.argmax(yf))
        peak_freq = xf[idx]
        if TONE_HZ - HZ_TOLERANCE <= peak_freq <= TONE_HZ + HZ_TOLERANCE:
            start_sec = start / samplerate
            end_sec = (start + window_size) / samplerate
            results.append({
                "type": "1KHZ_TONE",
                "start": to_hms(start_sec),
                "end": to_hms(end_sec),
                "duration": round(end_sec - start_sec, 2),
                "details": "1kHz audio tone"
            })
            break

    try:
        os.remove(TEMP_AUDIO)
    except Exception:
        pass
    print("   ✅ 1kHz tone detect done", flush=True)
    return results


def detect_ruis_gray_stripes(filepath,
                             fps_sample=RUIS_FPS_SAMPLE,
                             sat_max=RUIS_SAT_MAX,
                             lap_var_min=RUIS_LAP_VAR_MIN,
                             stripe_std_min=RUIS_STRIPE_STD_MIN,
                             min_duration=MIN_GLITCH_DURATION):
    """
    Серый экран с шумом/полосами (VHS-ruis/strepen):
    - Низкая насыщенность (серость)
    - Шумность/полосатость по Лапласиану или std столбцов
    """
    results = []
    cap = cv2.VideoCapture(filepath)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fps <= 0 or frames <= 0:
        cap.release()
        return results

    step = max(1, int(round(fps / max(0.1, fps_sample))))
    in_ruis = False
    seg_start_t = 0.0

    def frame_score(img_bgr):
        # downscale до высоты ~480 для скорости
        h0, w0 = img_bgr.shape[:2]
        scale = 480.0 / max(1, h0)
        if scale < 1.0:
            img_bgr = cv2.resize(img_bgr, (int(w0*scale), int(h0*scale)), interpolation=cv2.INTER_AREA)

        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        s_mean = float(hsv[..., 1].mean())

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
        lap_var = float(lap.var())

        col_mean = gray.mean(axis=0)
        stripe_std = float(col_mean.std())

        is_gray = s_mean <= sat_max
        noisy_or_striped = (lap_var >= lap_var_min) or (stripe_std >= stripe_std_min)
        return is_gray and noisy_or_striped, s_mean, lap_var, stripe_std

    for i in tqdm(range(0, frames, step),
                  desc=f"   📺 RUIS/STRIPES {os.path.basename(filepath)}",
                  unit="smp",
                  leave=False):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok:
            break
        t = i / fps

        flag, _, _, _ = frame_score(frame)

        if flag and not in_ruis:
            in_ruis = True
            seg_start_t = t
        elif not flag and in_ruis:
            dur = t - seg_start_t
            if dur >= min_duration:
                results.append({
                    "type": "RUIS/STRIPES",
                    "start": to_hms(seg_start_t),
                    "end": to_hms(t),
                    "duration": dur,
                    "details": f"gray+noisy/striped (S≤{sat_max}, lapVar≥{lap_var_min} or stripeSTD≥{stripe_std_min})"
                })
            in_ruis = False

    # если сегмент тянется до конца
    if in_ruis:
        t = frames / fps
        dur = t - seg_start_t
        if dur >= min_duration:
            results.append({
                "type": "RUIS/STRIPES",
                "start": to_hms(seg_start_t),
                "end": to_hms(t),
                "duration": dur,
                "details": "gray+noisy/striped (end)"
            })

    cap.release()
    return results


# =======================
#     MAIN PIPELINE
# =======================
def main():
    # Подготовим CSV
    with open(OUTPUT_CSV_EVENTS, mode='w', newline='') as events_csv, \
         open(OUTPUT_CSV_SUMMARY, mode='w', newline='') as summary_csv:

        events_writer = csv.writer(events_csv)
        summary_writer = csv.writer(summary_csv)

        events_writer.writerow(['video_file', 'type', 'start_time', 'end_time', 'duration_sec', 'details'])
        summary_writer.writerow(['video_file', 'video_duration_sec', 'video_duration_mmss',
                                 'errors_count', 'errors_total_sec', 'errors_total_mmss', 'damage_percent'])

        # Очередь: натурально отсортированные видео
        video_files = sorted(
            [f for f in os.listdir(VIDEO_FOLDER) if f.lower().endswith(VIDEO_EXTS)],
            key=natural_sort_key
        )

        for filename in tqdm(video_files, desc="📦 Videos", unit="file"):
            filepath = os.path.join(VIDEO_FOLDER, filename)
            print(f"\n🎨 Start analyse van {filename}...", flush=True)

            # длительность видео
            video_duration = get_video_duration_seconds(filepath)

            # собираем все результаты
            all_results = []
            all_results += detect_black_segments(filepath)
            all_results += detect_glitches(filepath)            # цветовые глитчи
            all_results += detect_freezes(filepath)             # фризы
            all_results += detect_1khz_tone(filepath)           # 1 kHz
            all_results += detect_ruis_gray_stripes(filepath)   # серый шум/полосы

            print(f"▶️ Verwerken: {filename}")
            if all_results:
                print(f"📄 Resultaten: {len(all_results)} fouten gevonden")

                total_defect_sec = 0.0
                for r in all_results:
                    total_defect_sec += float(r['duration'])
                    print(f"🧾 {r['type']} → {r['start']} → {r['end']} ({float(r['duration']):.2f} sec)")
                    events_writer.writerow([
                        filename, r["type"], r["start"], r["end"],
                        round(float(r["duration"]), 2), r["details"]
                    ])

                # Форматы для CSV (mm:ss) как раньше
                total_mmss = seconds_to_mmss(total_defect_sec)
                dur_mmss = seconds_to_mmss(video_duration) if video_duration > 0 else "00:00"

                # Новые форматы для печати (hh:mm:ss)
                total_hms = to_hms(total_defect_sec)
                video_hms = to_hms(video_duration) if video_duration > 0 else "00:00:00"

                # === ВАЖНО: считаем покрытие таймлайна (объединённые интервалы) ===
                intervals = [(hms_to_seconds(r['start']), hms_to_seconds(r['end'])) for r in all_results]
                merged = merge_intervals(intervals)
                covered_sec = sum(e - s for s, e in merged)

                damage_percent = (covered_sec / video_duration * 100.0) if video_duration > 0 else 0.0

                # Финальная строка: сумма длительностей + % по покрытию
                print(
                    f"📊 Totaal: {len(all_results)} fouten, totaalduur {total_hms} "
                    f"(= {int(round(total_defect_sec))} sec) — Video {video_hms}; "
                    f"Beschadiging: {damage_percent:.2f}%"
                )

                # CSV-сводка без изменения структуры
                summary_writer.writerow([
                    filename,
                    round(video_duration, 2),
                    dur_mmss,
                    len(all_results),
                    round(total_defect_sec, 2),
                    total_mmss,
                    round(damage_percent, 2)
                ])
            else:
                print("📄 Geen fouten gevonden.")
                dur_mmss = seconds_to_mmss(video_duration) if video_duration > 0 else "00:00"
                summary_writer.writerow([
                    filename,
                    round(video_duration, 2),
                    dur_mmss,
                    0, 0.0, "00:00", 0.0
                ])

    print(f"\n✅ Klaar! Подробный CSV: {OUTPUT_CSV_EVENTS}\n✅ Сводка по видео: {OUTPUT_CSV_SUMMARY}", flush=True)


if __name__ == "__main__":
    main()
