import os
import json
import numpy as np
import subprocess
import gc
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))

VIDEO_FILE = os.path.join(current_dir, "test_meeting.mp4")
EXTRACTED_AUDIO = os.path.join(current_dir, "extracted_audio.wav")
FINAL_CONSPECT_MD = os.path.join(current_dir, "FINAL_CORPORATE_CONSPECT.md")

def extract_audio_from_video(mp4_path, target_wav_path):
    print(f"[Конвейер] Системное извлечение аудиодорожки из {mp4_path}")
    command = [
        "ffmpeg", "-y", 
        "-i", mp4_path, 
        "-ar", "16000", 
        "-ac", "1", 
        "-c:a", "pcm_s16le", 
        target_wav_path
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    print(f"[Конвейер] Звук успешно извлечен и сохранен в: {target_wav_path}")

def run_production_orchestrator():
    print("="*70)
    print("🚀 ЗАПУСК ПОЛНОГО МУЛЬТИМОДАЛЬНОГО ИИ-ОРКЕСТРА (ВИДЕО + ЗВУК + ИИ)")
    print("="*70)

    # Шаг 0: Извлечение звука через FFmpeg
    extract_audio_from_video(VIDEO_FILE, EXTRACTED_AUDIO)

    # Шаг 1: Работа OpenCV (Не требует GPU)
    print("\n[ЭТАП 1] Запуск OpenCV: кадрирование и дедупликация кадров...")
    import VideoProcessor
    video_proc = VideoProcessor.CorporateVideoProcessor(mse_threshold=3.0, frame_skip=30)
    video_proc.extract_unique_slides(VIDEO_FILE)
    del video_proc

    # Шаг 2: ЛОКАЛЬНЫЙ ИМПОРТ И ЗАПУСК WHISPER
    print("\n[ЭТАП 2] Активация речевой модели Whisper...")
    import SpeechTranscriber
    speech_transcriber = SpeechTranscriber.CorporateSpeechTranscriber()
    real_speech_text = speech_transcriber.transcribe_audio(EXTRACTED_AUDIO)

    # Очистка VRAM после Whisper
    print("\n[Оптимизация] Полная выгрузка Whisper из VRAM...")
    del speech_transcriber
    del SpeechTranscriber
    import sys
    if 'SpeechTranscriber' in sys.modules: del sys.modules['SpeechTranscriber']
    gc.collect()
    torch.cuda.empty_cache()

    # Шаг 3: ЛОКАЛЬНЫЙ ИМПОРТ И ЗАПУСК QWEN-VL
    print("\n[ЭТАП 3] Активация Vision-Language модели Qwen-VL...")
    import SlideAnalyzer
    vlm_analyzer = SlideAnalyzer.CorporateSlideAnalyzer()
    slides_data = vlm_analyzer.analyze_all_extracted_slides()

    # Очистка VRAM после Qwen-VL
    print("\n[Оптимизация] Полная выгрузка Qwen-VL из VRAM...")
    del vlm_analyzer
    del SlideAnalyzer
    if 'SlideAnalyzer' in sys.modules: del sys.modules['SlideAnalyzer']
    gc.collect()
    torch.cuda.empty_cache() # Освобождает все 24 ГБ под финальный шаг!

    # Шаг 4: ЛОКАЛЬНЫЙ ИМПОРТ И ЗАПУСК ИИ-РЕДАКТОРА
    print("\n[ЭТАП 4] Передача данных Агенту-Редактору Qwen-7B...")
    import ConspectEditor
    editor = ConspectEditor.CorporateConspectEditor()
    
    slides_json_str = json.dumps(slides_data, ensure_ascii=False, indent=2)
    
    premium_conspect_markdown = editor.generate_human_style_conspect(
        whisper_text=real_speech_text, 
        vlm_slides_json=slides_json_str
    )

    # 1. Срезаем технические теги markdown по краям
    premium_conspect_markdown = premium_conspect_markdown.strip()
    if premium_conspect_markdown.startswith("```markdown"):
        premium_conspect_markdown = premium_conspect_markdown[11:].strip()
    if premium_conspect_markdown.endswith("```"):
        premium_conspect_markdown = premium_conspect_markdown[:-3].strip()

    # 2. Восстанавливаем префиксы путей к картинкам слайдов
    import re
    premium_conspect_markdown = re.sub(
        r'!\[\[(?!extracted_slides/)([^\]]+)\]\]', 
        r'![[extracted_slides/\1]]', 
        premium_conspect_markdown
    )

    # 3. Переводим разделители скобок в доллары
    premium_conspect_markdown = premium_conspect_markdown.replace(r"\(", "$").replace(r"\)", "$")
    premium_conspect_markdown = premium_conspect_markdown.replace(r"\[", "$$").replace(r"\]", "$$")

    # 4. АППАРАТНОЕ УДАЛЕНИЕ ПРОБЕЛОВ В ФОРМУЛАХ (ФИКС БАГА):
    # Убираем пробел ПОСЛЕ открывающего знака $ или $$
    premium_conspect_markdown = re.sub(r'(\$\$?)\s+', r'\1', premium_conspect_markdown)
    # Убираем пробел ПЕРЕД закрывающим знаком $ или $$
    premium_conspect_markdown = re.sub(r'\s+(\$\$?)', r'\1', premium_conspect_markdown)

    with open(FINAL_CONSPECT_MD, "w", encoding="utf-8") as f:
        f.write(premium_conspect_markdown)

    print("="*70)
    print("🎉 ОРКЕСТРАТОР СИСТЕМЫ УСПЕШНО ЗАВЕРШИЛ РАБОТУ!")
    print(f"ИДЕАЛЬНЫЙ КОНСПЕКТ СОХРАНЕН В: {FINAL_CONSPECT_MD}")

if __name__ == "__main__":
    run_production_orchestrator()
