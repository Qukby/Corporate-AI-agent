import os
import json
import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

current_dir = os.path.dirname(os.path.abspath(__file__))
device_str = "cuda" if torch.cuda.is_available() else "cpu"
PROMPTS_FILE = os.path.join(current_dir, "prompts.json")

class CorporateTestGenerator:
    def __init__(self):
        print("[Экзаменатор] Локальная загрузка Qwen-7B для генерации тестов...")
        self.model_id = "Qwen/Qwen2.5-7B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        # ДОБАВИЛИ КВАНТОВАНИЕ: защищает от нехватки памяти во время длинной генерации JSON
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, 
            torch_dtype="auto",
            quantization_config=quantization_config,
            device_map="auto"
        )

        config_json_path = os.path.join(current_dir, "prompts.json")
        with open(config_json_path, "r", encoding="utf-8") as f:
            config_paths = json.load(f)
            prompt_md_relative_path = config_paths["test_generator"]

        absolute_prompt_path = os.path.join(current_dir, prompt_md_relative_path)
        with open(absolute_prompt_path, "r", encoding="utf-8") as f:
            self.system_prompt = f.read()
            
        print("[Редактор] Промпт успешно загружен из внешнего .md файла!")
        
    def generate_questions_from_conspect(self, conspect_path: str) -> str:
        """
        Читает готовый конспект и генерирует 10 уникальных вопросов
        """
        if not os.path.exists(conspect_path):
            return "Файл конспекта не найден."

        with open(conspect_path, "r", encoding="utf-8") as f:
            conspect_content = f.read()

        user_content = f"""Вот готовый учебный конспект лекции:\n{conspect_content}\n\nСоставь по нему 10 уникальных вопросов для сотрудников:"""

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content}
        ]

        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([prompt], return_tensors="pt").to(device_str)
        
        with torch.no_grad():  
            generated_ids = self.model.generate(
                **inputs, 
                max_new_tokens=1536, 
                do_sample=True,
                temperature=0.2,
                top_p=0.9
            )
            
        generated_ids_trimmed = [oid[len(iids):] for iids, oid in zip(inputs.input_ids, generated_ids)]
        raw_output = self.tokenizer.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0].strip()

        json_match = re.search(r'\[.*\]', raw_output, re.DOTALL)
        if json_match:
            return json_match.group(0).strip()
            
        return raw_output

if __name__ == "__main__":
    # Короткий автономный тест модуля
    generator = CorporateTestGenerator()
    CONSPECT_FILE = os.path.join(current_dir, "FINAL_CORPORATE_CONSPECT.md")
    
    if os.path.exists(CONSPECT_FILE):
        questions = generator.generate_questions_from_conspect(CONSPECT_FILE)
        print("\n=== СПИСОК СГЕНЕРИРОВАННЫХ ВОПРОСОВ ОТ ИИ ===")
        print(questions)
        print("=============================================")