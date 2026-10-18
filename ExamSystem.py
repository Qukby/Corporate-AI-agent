import os
import json
import gc
import torch
import sys 
from datetime import datetime

current_dir = os.path.dirname(os.path.abspath(__file__))

# Пути к файлам в нашей системе
CONSPECT_FILE = os.path.join(current_dir, "FINAL_CORPORATE_CONSPECT.md")
QUESTIONS_JSON_FILE = os.path.join(current_dir, "generated_questions.json")

USER_RESULT_FILE = os.path.join(current_dir, "user_test_result.json")

def run_interactive_examination():
    print("="*70)
    print("🎓 ЗАПУСК ИНТЕРАКТИВНОЙ ЭКЗАМЕНАЦИОННОЙ СИСТЕМЫ ПРОВЕРКИ ЗНАНИЙ")
    print("="*70)

    # -----------------------------------------------------------------
    # ЭТАП 1: Получение списка вопросов (Генерация или чтение кэша)
    # -----------------------------------------------------------------
    if not os.path.exists(QUESTIONS_JSON_FILE):
        print("\n[Система] Файл вопросов не найден. Запуск ИИ-Генератора тестов...")
        
        # Локально инициализируем генератор вопросов
        import TestGenerator
        test_gen = TestGenerator.CorporateTestGenerator()
        raw_questions_string = test_gen.generate_questions_from_conspect(CONSPECT_FILE)
        
        # ЖЕСТКАЯ ОЧИСТКА VRAM: выгружаем веса генератора полностью
        print("\n[Оптимизация] Выгружаем генератор вопросов и очищаем VRAM...")
        del test_gen
        del TestGenerator
        if 'TestGenerator' in sys.modules: 
            del sys.modules['TestGenerator']
        gc.collect()
        torch.cuda.empty_cache()

        # Фильтруем строку, если модель добавила лишние markdown-теги json
        raw_questions_string = raw_questions_string.strip()
        if raw_questions_string.startswith("```json"):
            raw_questions_string = raw_questions_string[7:].strip()
        if raw_questions_string.endswith("```"):
            raw_questions_string = raw_questions_string[:-3].strip()

        # Сохраняем сгенерированные вопросы в файл для бэкенда будущего сайта
        try:
            questions_list = json.loads(raw_questions_string)
            with open(QUESTIONS_JSON_FILE, "w", encoding="utf-8") as f:
                json.dump(questions_list, f, ensure_ascii=False, indent=4)
            print(f"[Система] Создан новый пул вопросов и сохранен в: {QUESTIONS_JSON_FILE}")
        except json.JSONDecodeError:
            print("[Критическая Ошибка] Нейросеть выдала невалидный JSON. Попробуйте запустить заново.")
            return
    else:
        print(f"\n[Система] Обнаружен готовый пул вопросов на диске! Загрузка из {QUESTIONS_JSON_FILE}...")
        with open(QUESTIONS_JSON_FILE, "r", encoding="utf-8") as f:
            questions_list = json.load(f)

    # -----------------------------------------------------------------
    # ЭТАП 2: Инициализация ИИ-Судьи (Оценщика ответов)
    # -----------------------------------------------------------------
    print("\n[Система] Запуск ИИ-Оценщика ответов (AnswerJudge)...")
    import AnswerJudge
    ai_judge = AnswerJudge.CorporateAnswerJudge()

    # -----------------------------------------------------------------
    # ЭТАП 3: Интерактивный интерактивный опрос в консоли 
    # -----------------------------------------------------------------
    print(f"\n[Старт] В базе доступно {len(questions_list)} вопросов. Начинаем экзамен!")
    print("-"*70)

    detailed_answers_history = []
    total_score_sum = 0

    # Проверим сотрудника на первых двух вопросах из списка (можно заменить на весь список)
    for idx, item in enumerate(questions_list):
        print(f"\n📝 [ВОПРОС №{idx + 1} Тип: {item.get('type', 'Свободный')}]:")
        print(f"👉 {item['question']}")
        
        if "options" in item:
            print("Варианты ответов:")
            for opt in item["options"]:
                print(f"  - {opt}")

        # Просим пользователя ввести развернутый ответ в консоли
        user_answer = input("\n✍️  Введите ваш ответ для ИИ-Профессора: ").strip()
        
        if not user_answer:
            print("[Система] Ответ не введен. Автоматически выставляется 0 баллов.")
            user_answer = "Ответ отсутствует."
            score = 0
            verdict = "Сотрудник проигнорировал вопрос и не ввел текстовый ответ."
        else:
            print("\n[ИИ-Судья] Проверка ответа сотрудника, сопоставление с конспектом...")
        
            # Отправляем вопрос и ответ человека в наш ИИ-судейский модуль
            judgment = ai_judge.evaluate_employee_answer(
            question=item['question'], 
            employee_answer=user_answer
            )
            score = int(judgment.get("score", 0))
            verdict = judgment.get("verdict", "Нет комментария.")

        print("\n" + "🎓 Вердикт Проверяющего:")
        print(f"➡️  Текущая оценка: {score} / 10 баллов")
        print(f"➡️  Рецензия: {verdict}")
        print("-"*70)

        # Копим общую сумму баллов для вычисления среднего
        total_score_sum += score
        
        # Бережно упаковываем детали ответа в наш архив
        detailed_answers_history.append({
            "question_num": idx + 1,
            "question_text": item['question'],
            "employee_answer": user_answer,
            "ai_score": score,
            "ai_verdict": verdict
        })

    # -----------------------------------------------------------------
    # ЭТАП 4: Подсчет финального рейтинга и ЖЕСТКОЕ СОХРАНЕНИЕ В JSON
    # -----------------------------------------------------------------
    # Вычисляем среднее арифметическое за весь тест (округляем до сотых)
    final_average_score = round(total_score_sum / len(questions_list), 2) if questions_list else 0
    
    # Формируем красивую структуру личного дела сотрудника
    final_report_payload = {
        "employee_username": "worker3000", # Имя пользователя для профиля сайта
        "exam_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_average_score": final_average_score,
        "test_history": detailed_answers_history
    }

    # Записываем отчет на жесткий диск сервера в файл user_test_result.json
    print("\n[Сохранение] Фиксация результатов экзамена в личном деле сотрудника...")
    with open(USER_RESULT_FILE, "w", encoding="utf-8") as f:
        json.dump(final_report_payload, f, ensure_ascii=False, indent=4)

    print("\n" + "="*70)
    print("🎉 ЭКЗАМЕН ПОЛНОСТЬЮ ЗАВЕРШЕН И ЗАФИКСИРОВАН НА ДИСКЕ!")
    print(f"Итоговый средний балл сотрудника: {final_average_score} / 10")
    print(f"Детальный лог ответов сохранен в: {USER_RESULT_FILE}")
    print("="*70)

if __name__ == "__main__":
    run_interactive_examination()
