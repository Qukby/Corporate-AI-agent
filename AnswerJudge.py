import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

current_dir = os.path.dirname(os.path.abspath(__file__))
device_str = "cuda" if torch.cuda.is_available() else "cpu"
PROMPTS_FILE = os.path.join(current_dir, "prompts.json")
CONSPECT_FILE = os.path.join(current_dir, "FINAL_CORPORATE_CONSPECT.md")

class CorporateAnswerJudge:
    def __init__(self):
        print("[Судья] Локальная загрузка Qwen-7B для оценки ответов...")
        self.model_id = "Qwen/Qwen2.5-7B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        
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
        
        # Читаем путь к промпту судьи из prompts.json
        with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
            config_paths = json.load(f)
            prompt_md_relative_path = config_paths["answer_judge"]

        # Загружаем сам текст промпта из .md документа
        absolute_prompt_path = os.path.join(current_dir, prompt_md_relative_path)
        with open(absolute_prompt_path, "r", encoding="utf-8") as f:
            self.system_prompt = f.read()
            
        print("[Судья] Системный промпт успешно загружен из внешнего .md файла!")

    def evaluate_employee_answer(self, question: str, employee_answer: str) -> dict:
        """
        Сопоставляет ответ сотрудника с конспектом и возвращает оценку + вердикт в JSON
        """
        if not os.path.exists(CONSPECT_FILE):
            return {"error": "Файл конспекта не найден. Сначала запустите MainOrchestrator.py"}

        with open(CONSPECT_FILE, "r", encoding="utf-8") as f:
            conspect_content = f.read()

        # Формируем пакет документов для ИИ-Судьи
        user_content = f"""Вот эталонный конспект лекции:
{conspect_content}

---
Экзаменационный вопрос: {question}
Ответ сотрудника: {employee_answer}

Вынеси строгое решение в формате JSON:"""

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content}
        ]
        
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([prompt], return_tensors="pt").to(device_str)
        
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=512, temperature=0.1) # Низкая температура для строгой оценки
            
        generated_ids_trimmed = [oid[len(iids):] for iids, oid in zip(inputs.input_ids, generated_ids)]
        raw_output = self.tokenizer.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0].strip()

        # Аппаратный фильтр: если модель случайно добавила теги ```json ... ```, срезаем их
        if raw_output.startswith("```json"):
            raw_output = raw_output[7:].strip()
        if raw_output.endswith("```"):
            raw_output = raw_output[:-3].strip()

        # Парсим строку в реальный словарь Python, чтобы бэкенд сайта мог сразу забрать её в базу данных
        try:
            return json.loads(raw_output)
        except json.JSONDecodeError:
            # Если модель нарушила структуру, возвращаем сырой текст как вердикт
            return {"score": 0, "verdict": raw_output}

if __name__ == "__main__":
    # Быстрый тест работы нашего ИИ-Судьи
    judge = CorporateAnswerJudge()
    
    # Имитируем ситуацию: вопрос из теста и плохой ответ сотрудника
    test_question = "Какое каноническое уравнение имеет эллиптический цилиндр?"
    test_answer = "Ну это такая круглая труба, которая идет вверх по оси Z, формулу точно не помню, но там что-то с плюсом."
    
    print("\n[Тест] Отправляем ответ сотрудника на проверку ИИ-Профессору...")
    result_json = judge.evaluate_employee_answer(test_question, test_answer)
    
    print("\n=== ИТОГОВЫЙ СУДЕБНЫЙ ВЕРДИКТ В ФОРМАТЕ JSON ДЛЯ САЙТА ===")
    print(json.dumps(result_json, ensure_ascii=False, indent=2))
    print("==========================================================")
