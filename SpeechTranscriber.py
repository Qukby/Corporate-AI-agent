import os
import torch
from transformers import pipeline, BitsAndBytesConfig

current_dir = os.path.dirname(os.path.abspath(__file__))
device_str = "cuda" if torch.cuda.is_available() else "cpu"

# 4-битное квантование для экономии VRAM на сервере (чтобы Whisper и Qwen-VL поместились вместе!)
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

print("[Whisper] Загрузка модели распознавания речи openai/whisper-tiny...")
# Инициализируем пайплайн автоматического распознавания речи (ASR)
asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-tiny",
    device_map="auto",
    model_kwargs={"quantization_config": quantization_config} # Сжимаем Whisper
)

class CorporateSpeechTranscriber:
    def transcribe_audio(self, audio_path: str) -> str:
        """
        Принимает путь к .wav файлу и возвращает чистый текст
        """
        if not os.path.exists(audio_path):
            return "Аудиофайл не найден для транскрибации."
            
        print(f"[Whisper] Запуск распознавания речи для файла: {audio_path}...")
        
        # Запускаем инференс. Параметр generate_kwargs указывает язык распознавания
        result = asr_pipeline(
            audio_path, 
            # Добавили return_timestamps=True, чтобы Whisper умел обрабатывать файлы длиннее 30 секунд
            return_timestamps=True, 
            generate_kwargs={
                "language": "russian",
                "task": "transcribe"
            }
        )
        
        print("[Whisper] Речь успешно переведена в текст!")
        return result["text"].strip()

if __name__ == "__main__":
    # Тест модуля
    transcriber = CorporateSpeechTranscriber()
    TEST_WAV = os.path.join(current_dir, "test_call.wav")
    if os.path.exists(TEST_WAV):
        text = transcriber.transcribe_audio(TEST_WAV)
        print(f"\nРезультат теста Whisper:\n{text}")