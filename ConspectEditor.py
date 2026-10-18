import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

current_dir = os.path.dirname(os.path.abspath(__file__))
device_str = "cuda" if torch.cuda.is_available() else "cpu"

class CorporateConspectEditor:
    def __init__(self):
        print("[Редактор] Загрузка Qwen-7B для сборки конспекта...")
        model_id = "Qwen/Qwen2.5-7B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype="auto",
            device_map=device_str
        )

        # 1. Читаем карту путей из центрального JSON
        config_json_path = os.path.join(current_dir, "prompts.json")
        with open(config_json_path, "r", encoding="utf-8") as f:
            config_paths = json.load(f)
            # Вытаскиваем относительный адрес .md файла промпта
            prompt_md_relative_path = config_paths["conspect_editor"]

        # 2. Считываем чистый текст промпта из самого .md документа
        absolute_prompt_path = os.path.join(current_dir, prompt_md_relative_path)
        with open(absolute_prompt_path, "r", encoding="utf-8") as f:
            self.system_prompt = f.read()
            
        print("[Редактор] Промпт успешно загружен из внешнего .md файла!")

    def generate_human_style_conspect(self, whisper_text: str, vlm_slides_json: str) -> str:
        """
        Берет сырые данные и превращает их в красивую Obsidian книгу
        """
        
        user_content = f"""Вот сырая стенограмма речи лектора:
        {whisper_text}

        Вот сырые описания и имена файлов вырезанных слайдов презентации:
        {vlm_slides_json}

        Собери из этого идеальный интерактивный учебный конспект в Markdown:"""

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content}
        ]
        
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([prompt], return_tensors="pt").to(device_str)
        
        with torch.no_grad():
            # Даем большой лимит токенов (2048), чтобы получилась полноценная книга-конспект
            generated_ids = self.model.generate(**inputs, max_new_tokens=2048, temperature=0.3)
            
        generated_ids_trimmed = [oid[len(iids):] for iids, oid in zip(inputs.input_ids, generated_ids)]
        return self.tokenizer.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0].strip()