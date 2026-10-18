import os
import cv2
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
OUTPUT_SLIDES_DIR = os.path.join(current_dir, "extracted_slides")
os.makedirs(OUTPUT_SLIDES_DIR, exist_ok=True)

class CorporateVideoProcessor:
    def __init__(self, mse_threshold: float = 3.0, frame_skip: int = 30):
        """
        :param mse_threshold: Порог чувствительности. Чем ниже, тем деликатнее ловит изменения.
        :param frame_skip: Сколько кадров пропускать. Проверять каждый кадр нет смысла. 
                           Если видео 30 FPS, frame_skip=30 означает проверку 1 раз в секунду.
        """
        self.mse_threshold = mse_threshold
        self.frame_skip = frame_skip

    def calculate_mse(self, imageA: np.ndarray, imageB: np.ndarray) -> float:
        """
        Низкоуровневый подсчет среднеквадратичной ошибки между двумя матрицами пикселей.
        OpenCV и NumPy выполняют это на C++ скоростях.
        """
        # Превращаем в float, чтобы при вычитании не было переполнения uint8
        err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
        err /= float(imageA.shape[0] * imageA.shape[1])
        return err

    def extract_unique_slides(self, video_path: str):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Видеофайл не найден: {video_path}")

        print(f"[Видео] Открываем видеопоток: {video_path}")
        # Инициализируем C++ объект захвата видео OpenCV
        cap = cv2.VideoCapture(video_path)

        # Получаем ширину и высоту
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        print(f"[Видео] Успешно! FPS звонка: {fps:.2f}, Всего кадров: {int(total_frames)}")

        prev_frame_gray = None
        saved_slides_count = 0
        frame_idx = 0

        while cap.isOpened():
            # Заставляем курсор OpenCV прыгнуть сразу на нужный кадр (наша skip-оптимизация)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if not ret:
                break # Видео закончилось

            # Вычисляем текущую секунду звонка
            current_second = int(frame_idx / fps)

            # Переводим кадр в черно-белый (Grayscale). 
            # Для отлова слайдов цвет не важен, а память и скорость возрастают в 3 раза!
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Размываем картинку (Gaussian Blur), чтобы убрать мелкие шумы сжатия видео/Zoom
            gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)

            # Допустим, разрешение кадра 1920x1080. 
            # Отрезаем правые 400 пикселей (где камера) и смотрим только на левую часть:
            gray_frame_cropped = gray_frame[0:height, 0:width-400]

            if prev_frame_gray is not None:
                # Считаем разницу между текущим слайдом и предыдущим сохраненным
                mse_score = self.calculate_mse(gray_frame_cropped, prev_frame_gray)

                # Если ошибка превысила порог — значит слайд перелистнули!
                if mse_score > self.mse_threshold:
                    saved_slides_count += 1
                    slide_filename = os.path.join(OUTPUT_SLIDES_DIR, f"slide_sec_{current_second}.png")

                # Сохраняем ОРИГИНАЛЬНЫЙ цветной кадр на диск
                cv2.imwrite(slide_filename, frame)
                print(f"==> [Отлов слайда] Слайд сменился на {current_second}-й секунде! (MSE: {mse_score:.2f}). Сохранено.")

                # Обновляем эталон для сравнения
                prev_frame_gray = gray_frame_cropped

            else:
                # Это самый первый кадр видео (Старт презентации). Сохраняем его железно.
                saved_slides_count += 1
                slide_filename = os.path.join(OUTPUT_SLIDES_DIR, f"slide_sec_{current_second}.png")
                cv2.imwrite(slide_filename, frame)
                print(f"==> [Старт] Сохранен стартовый слайд звонка на {current_second}-й секунде.")
                prev_frame_gray = gray_frame_cropped

            # Двигаем индекс вперед на 1 секунду времени видео
            frame_idx += self.frame_skip

        cap.release()
        print(f"\n[Видео] Анализ завершен! Вырезано уникальных слайдов: {saved_slides_count}")

if __name__ == "__main__":
    print("=== ИНИЦИАЛИЗАЦИЯ КОРПОРАТИВНОГО ВИДЕО-ПРОЦЕССОРА ===")
    
    # Для теста укажи путь к любому своему .mp4 файлу созвона/записи экрана
    # Замени "test_meeting.mp4" на реальное имя твоего видеофайла
    TEST_VIDEO = os.path.join(current_dir, "test_meeting.mp4")
    
    # Давай сделаем проверку, чтобы код не падал, если видео еще нет
    if not os.path.exists(TEST_VIDEO):
        print(f"[Предупреждение] Положи любой видеофайл .mp4 под именем 'test_meeting.mp4' в папку: {current_dir}")
    else:
        processor = CorporateVideoProcessor(mse_threshold=3.0, frame_skip=30)
        processor.extract_unique_slides(TEST_VIDEO)