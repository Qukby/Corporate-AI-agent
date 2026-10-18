import warnings
# Жестко приказываем Python игнорировать любые предупреждения о депрекации (устаревании кода)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import torch
import json
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

current_dir = os.path.dirname(os.path.abspath(__file__))
SLIDES_DIR = os.path.join(current_dir, "extracted_slides")

device_str = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[VLM] Используем устройство: {device_str}")

class CorporateSlideAnalyzer:
    def __init__(self):
        config_json_path = os.path.join(current_dir, "prompts.json")
        with open(config_json_path, "r", encoding="utf-8") as f:
            config_paths = json.load(f)
            # Вытаскиваем относительный адрес .md файла промпта
            prompt_md_relative_path = config_paths["vlm_slide_analyzer"]

        absolute_prompt_path = os.path.join(current_dir, prompt_md_relative_path)
        with open(absolute_prompt_path, "r", encoding="utf-8") as f:
            self.system_prompt = f.read()

        print("[Редактор] Промпт успешно загружен из внешнего .md файла!")
            
        # Мы берем самую компактную и быструю версию 2B, чтобы она летала локально
        model_id = "Qwen/Qwen2-VL-7B-Instruct"

        print(f"[VLM] Загрузка Сжатой Vision-модели {model_id}...")
        self.processor = AutoProcessor.from_pretrained(model_id)

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, # Используем стандартную точность 16-бит
            device_map="auto"
        )

    def analyze_single_slide(self, image_path: str) -> str:
        """
        Прогон одной картинки через Vision-нейросеть Qwen-VL
        """
        if not os.path.exists(image_path):
            return "Файл изображения не найден."

        # Формируем структуру сообщений, включая путь к картинке
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": self.system_prompt}
                ],
            }
        ]

        # Подготавливаем промпт и изображение через встроенные утилиты Hugging Face
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        # Превращаем всё в тензоры PyTorch и отправляем на видеокарту
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(device_str)

        # Запускаем прямой проход (Forward Inference) Vision-модели без вычисления градиентов
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=512)

        generated_ids_trimmed = [
            out_ids[len(input_ids):] for input_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return output_text[0].strip()

    def analyze_all_extracted_slides(self) -> dict:
        """
        Сканирует папку extracted_slides и анализирует каждый скриншот
        """
        slide_files = [f for f in os.listdir(SLIDES_DIR) if f.endswith(".png")]
        # Сортируем файлы по секундам в имени, чтобы конспект шел строго по хронологии созвона
        slide_files.sort(key=lambda x: int(x.split("_")[2].split(".")[0]))

        slides_descriptions = {}

        print(f"\n[VLM] Найдено {len(slide_files)} уникальных слайдов для анализа.")

        for slide in slide_files:
            full_path = os.path.join(SLIDES_DIR, slide)
            # Вытаскиваем секунду из имени файла (например, из 'slide_sec_45.png' вытащит 45)
            second = slide.split("_")[2].split(".")[0]

            print(f"[VLM] Анализируем слайд созвона на {second}-й секунде...")
            description = self.analyze_single_slide(full_path)

            slides_descriptions[second] = {
                "image_name": slide,
                "description": description
            }

        print("\n[VLM] Все слайды успешно обработаны нейросетью!")
        return slides_descriptions

if __name__ == "__main__":
    # Быстрый тест модуля
    analyzer = CorporateSlideAnalyzer()

    # Проверим, есть ли хоть одна картинка в папке после работы твоего видеопроцессора
    slide_files = [f for f in os.listdir(SLIDES_DIR) if f.endswith(".png")]
    if not slide_files:
        print(f"[Тест] Папка {SLIDES_DIR} пуста. Сначала запусти VideoProcessor.py на реальном .mp4 файле.")
    else:
        # Протестируем на самом первом вырезанном слайде
        first_slide_path = os.path.join(SLIDES_DIR, slide_files[0])
        print(f"[Тест] Запуск ИИ-анализа для файла: {slide_files[0]}")
        result = analyzer.analyze_single_slide(first_slide_path)
        print("\n=== РЕЗУЛЬТАТ ОПИСАНИЯ СЛАЙДА МОДЕЛЬЮ QWEN-VL ===")
        print(result)
        print("=================================================")