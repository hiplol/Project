# active_learning.py
import os
import csv
import threading
import pandas as pd


class ActiveLearner:
    def __init__(self,
                 classifier,
                 dataset_path: str = "code_dataset.csv",
                 uncertain_low: float = 40.0,
                 uncertain_high: float = 60.0,
                 min_samples_to_retrain: int = 5,
                 retrain_epochs: int = 30,
                 log_callback=print):
        self.classifier = classifier
        self.dataset_path = dataset_path
        self.uncertain_low = uncertain_low
        self.uncertain_high = uncertain_high
        self.min_samples_to_retrain = min_samples_to_retrain
        self.retrain_epochs = retrain_epochs
        self.log = log_callback

        self.buffer_X = []
        self.buffer_y = []
        self.lock = threading.Lock()
        self.is_retraining = False

    # ------------------------------------------------------------ helpers
    def is_uncertain(self, probability):
        if probability is None:
            return False
        return self.uncertain_low <= probability <= self.uncertain_high

    def add_example(self, code: str, label: int):
        if label not in (0, 1):
            self.log(f"⚠️ Пропущена невалидная метка: {label}")
            return
        if not code or not code.strip():
            self.log("⚠️ Пропущен пустой код")
            return

        with self.lock:
            self.buffer_X.append(code)
            self.buffer_y.append(int(label))
            n = len(self.buffer_X)
        self.log(f"➕ Добавлен пример в буфер активного обучения "
                 f"(метка={label}). В буфере: {n}")
        if n >= self.min_samples_to_retrain:
            self._trigger_retrain()

    # ------------------------------------------------------------ retrain
    def _trigger_retrain(self):
        with self.lock:
            if self.is_retraining:
                self.log("⏳ Дообучение уже идёт, пропускаю.")
                return
            self.is_retraining = True

        t = threading.Thread(target=self._retrain, daemon=True)
        t.start()

    def _retrain(self):
        try:
            self.log("🔁 Начинаю дообучение на новых примерах...")

            with self.lock:
                new_X = list(self.buffer_X)
                new_y = list(self.buffer_y)
                self.buffer_X.clear()
                self.buffer_y.clear()

            base_texts, base_labels = self._load_base_dataset()

            all_texts = list(base_texts) + new_X
            all_labels = list(base_labels) + new_y

            self.log(f"🔤 Переобучаю TF-IDF на {len(all_texts)} текстах...")
            x_vec, y_vec = self.classifier.preprocess_data(
                all_texts, all_labels, fit_vectorizer=True
            )
            new_dim = x_vec.shape[1]
            self.log(f"   Новая размерность вектора: {new_dim}")

            if self.classifier.model_type == 'nn':
                self.log(f"🧠 Пересобираю нейросеть под размерность {new_dim}...")
                self.classifier.train(
                    x_vec, y_vec,
                    epochs=self.retrain_epochs,
                    rebuild_nn=True,
                )
            else:
                self.log("🌲 Переобучаю RandomForest...")
                self.classifier.train(x_vec, y_vec, rebuild_nn=True)

            self.classifier.save_model()

            self._append_to_dataset(new_X, new_y)

            self.log(f"✅ Дообучение завершено на {len(new_X)} новых примерах. "
                     f"Модель сохранена.")

        except Exception as e:
            self.log(f"🔥 Ошибка дообучения: {e}")
        finally:
            with self.lock:
                self.is_retraining = False

    # ------------------------------------------------------------ dataset
    def _load_base_dataset(self):
        if not os.path.exists(self.dataset_path):
            self.log(f"⚠️ Датасет {self.dataset_path} не найден, "
                     f"дообучение будет только на новых примерах.")
            return [], []

        try:
            data = pd.read_csv(self.dataset_path, sep='~')
            if len(data.columns) != 2:
                raise ValueError("Ожидается 2 столбца")
            data.columns = ['code', 'label']
            return (data['code'].astype(str).tolist(),
                    data['label'].astype(int).tolist())
        except Exception as e:
            self.log(f"⚠️ Не удалось прочитать датасет: {e}")
            return [], []

    def _append_to_dataset(self, texts, labels):
        try:
            need_header = not os.path.exists(self.dataset_path)
            with open(self.dataset_path, "a", encoding="utf-8", newline="") as f:
                writer = csv.writer(f, delimiter='~')
                if need_header:
                    writer.writerow(["code", "label"])
                for t, y in zip(texts, labels):
                    writer.writerow([t, y])
            self.log(f"💾 Новые примеры добавлены в {self.dataset_path}")
        except Exception as e:
            self.log(f"⚠️ Не удалось записать в датасет: {e}")
