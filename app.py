# app.py
import os
import csv
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
import webbrowser

import customtkinter as ctk

import parser as parser_module
import my_ai
from active_learning import ActiveLearner


ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


# ===================================================== CodeReviewDialog
class CodeReviewDialog(ctk.CTkToplevel):
    def __init__(self, parent, code: str, prob: float, submission_data: dict):
        super().__init__(parent)
        self.title("Неопределённый случай — проверьте код")
        self.geometry("1100x760")
        self.minsize(800, 600)
        self.grab_set()
        self.result = None

        header = ctk.CTkFrame(self, corner_radius=8)
        header.pack(fill="x", padx=12, pady=(12, 6))

        ctk.CTkLabel(
            header,
            text=f"ID: {submission_data.get('id', '?')}   |   "
                 f"Задача: {submission_data.get('problem', '?')}",
            font=ctk.CTkFont(size=15, weight="bold"), anchor="w"
        ).pack(fill="x", padx=10, pady=(8, 2))

        ctk.CTkLabel(
            header,
            text=f"Автор: {submission_data.get('author', '?')}   |   "
                 f"Язык: {submission_data.get('lang', '?')}   |   "
                 f"Вердикт: {submission_data.get('verdict', '?')}",
            anchor="w"
        ).pack(fill="x", padx=10, pady=2)

        prob_color = "#e5c07b" if 40 <= prob <= 60 else "#98c379"
        ctk.CTkLabel(
            header, text=f"Вероятность списывания: {prob}%",
            text_color=prob_color,
            font=ctk.CTkFont(size=14, weight="bold"), anchor="w"
        ).pack(fill="x", padx=10, pady=(2, 8))

        code_frame = ctk.CTkFrame(self, corner_radius=8)
        code_frame.pack(fill="both", expand=True, padx=12, pady=6)

        ctk.CTkLabel(
            code_frame, text="Исходный код:",
            font=ctk.CTkFont(size=13, weight="bold"), anchor="w"
        ).pack(fill="x", padx=10, pady=(8, 2))

        # Показываем код
        self.text = ctk.CTkTextbox(
            code_frame, wrap="none",
            font=ctk.CTkFont(family="Consolas", size=12),
        )
        self.text.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        if not code or not code.strip():
            self.text.insert("1.0", "⚠️  Код пуст — возможно, посылка скрыта "
                                     "или не загрузилась.")
        else:
            # Добавляем номера строк для удобства
            lines = code.splitlines()
            width = len(str(len(lines)))
            numbered = "\n".join(
                f"{i:>{width}} | {line}" for i, line in enumerate(lines, 1)
            )
            self.text.insert("1.0", numbered)
        self.text.configure(state="disabled")

        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(fill="x", padx=12, pady=(0, 12))

        ctk.CTkButton(btn_frame, text="✗ Не списано",
                      fg_color="#3a7d3a", hover_color="#4a8d4a",
                      width=160, command=self._on_no).pack(side="left", padx=6)
        ctk.CTkButton(btn_frame, text="✓ Списано",
                      fg_color="#a03a3a", hover_color="#b04a4a",
                      width=160, command=self._on_yes).pack(side="left", padx=6)
        ctk.CTkButton(btn_frame, text="⏭ Пропустить",
                      fg_color="#555555", hover_color="#666666",
                      width=160, command=self._on_skip).pack(side="left", padx=6)

        self.bind("<Escape>", lambda e: self._on_skip())
        self.bind("<Control-y>", lambda e: self._on_yes())
        self.bind("<Control-n>", lambda e: self._on_no())

        self.update_idletasks()
        self._center(parent)

    def _center(self, parent):
        try:
            px, py = parent.winfo_rootx(), parent.winfo_rooty()
            pw, ph = parent.winfo_width(), parent.winfo_height()
            w, h = self.winfo_width(), self.winfo_height()
            self.geometry(f"{w}x{h}+{px + (pw - w)//2}+{py + (ph - h)//2}")
        except Exception:
            pass

    def _on_yes(self):
        self.result = 1
        self.destroy()

    def _on_no(self):
        self.result = 0
        self.destroy()

    def _on_skip(self):
        self.result = None
        self.destroy()


# ================================================================ App
class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Codeforces Antiplagiat")
        self.geometry("1400x850")
        self.minsize(1100, 700)

        self.classifier = None
        self.active_learner = None
        self.parser_thread = None
        self.train_thread = None
        self.stop_flag = False

        # Хранилища
        self.submissions = {}
        self.id_to_row = {}
        self.row_to_id = {}
        self.selected_id = None

        self._build_ui()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        header = ctk.CTkFrame(self, height=48, corner_radius=0)
        header.pack(fill="x")
        ctk.CTkLabel(
            header, text="  Codeforces Antiplagiat",
            font=ctk.CTkFont(size=18, weight="bold"), anchor="w"
        ).pack(side="left", padx=10, pady=8)

        main = ctk.CTkFrame(self, fg_color="transparent")
        main.pack(fill="both", expand=True, padx=12, pady=8)
        main.grid_columnconfigure(0, weight=0, minsize=360)
        main.grid_columnconfigure(1, weight=1)
        main.grid_rowconfigure(0, weight=1)

        # ---------------- LEFT COLUMN -------------------------------------
        left = ctk.CTkScrollableFrame(main, width=360)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        ctk.CTkLabel(left, text="Авторизация",
                     font=ctk.CTkFont(size=14, weight="bold")).pack(
            anchor="w", padx=8, pady=(8, 4))
        auth = ctk.CTkFrame(left, corner_radius=8)
        auth.pack(fill="x", padx=4, pady=4)
        ctk.CTkLabel(
            auth,
            text=("Авторизация через Chrome-профиль (папка chrome_profile).\n"
                  "При первом запуске откроется браузер — залогиньтесь "
                  "и пройдите Cloudflare.\nПрофиль сохранится."),
            justify="left", anchor="w", text_color="#a0a0a0", wraplength=320,
        ).pack(anchor="w", padx=10, pady=8)

        ctk.CTkLabel(left, text="Контест",
                     font=ctk.CTkFont(size=14, weight="bold")).pack(
            anchor="w", padx=8, pady=(12, 4))
        lf = ctk.CTkFrame(left, corner_radius=8)
        lf.pack(fill="x", padx=4, pady=4)
        ctk.CTkLabel(lf, text="URL или contestId:", anchor="w").pack(
            fill="x", padx=10, pady=(8, 2))
        self.url_var = tk.StringVar()
        ctk.CTkEntry(
            lf, textvariable=self.url_var,
            placeholder_text="https://codeforces.com/group/.../contest/643263"
        ).pack(fill="x", padx=10, pady=(0, 4))
        ctk.CTkLabel(
            lf, text="Пример: .../group/bVSVAj3Smd/contest/643263 или 643263",
            text_color="#808080", justify="left", anchor="w", wraplength=320,
        ).pack(fill="x", padx=10, pady=(0, 10))

        ctk.CTkLabel(left, text="Модель",
                     font=ctk.CTkFont(size=14, weight="bold")).pack(
            anchor="w", padx=8, pady=(12, 4))
        mf = ctk.CTkFrame(left, corner_radius=8)
        mf.pack(fill="x", padx=4, pady=4)

        ctk.CTkLabel(mf, text="Путь к модели:", anchor="w").pack(
            fill="x", padx=10, pady=(8, 2))
        row1 = ctk.CTkFrame(mf, fg_color="transparent")
        row1.pack(fill="x", padx=10, pady=(0, 6))
        self.model_path_var = tk.StringVar(value="code_classifier")
        ctk.CTkEntry(row1, textvariable=self.model_path_var).pack(
            side="left", fill="x", expand=True)
        ctk.CTkButton(row1, text="...", width=36,
                      command=self._choose_model_dir).pack(side="left", padx=(6, 0))

        ctk.CTkLabel(mf, text="Тип модели:", anchor="w").pack(
            fill="x", padx=10, pady=(0, 2))
        self.model_type_var = tk.StringVar(value="nn")
        ctk.CTkOptionMenu(mf, values=["nn", "rf"],
                          variable=self.model_type_var).pack(
            fill="x", padx=10, pady=(0, 6))

        ctk.CTkLabel(mf, text="Датасет (CSV):", anchor="w").pack(
            fill="x", padx=10, pady=(0, 2))
        row2 = ctk.CTkFrame(mf, fg_color="transparent")
        row2.pack(fill="x", padx=10, pady=(0, 6))
        self.dataset_path_var = tk.StringVar(value="code_dataset.csv")
        ctk.CTkEntry(row2, textvariable=self.dataset_path_var).pack(
            side="left", fill="x", expand=True)
        ctk.CTkButton(row2, text="...", width=36,
                      command=self._choose_dataset).pack(side="left", padx=(6, 0))

        ctk.CTkLabel(mf, text="Эпох обучения:", anchor="w").pack(
            fill="x", padx=10, pady=(0, 2))
        self.train_epochs_var = tk.StringVar(value="100")
        ctk.CTkEntry(mf, textvariable=self.train_epochs_var).pack(
            fill="x", padx=10, pady=(0, 10))

        self.load_train_btn = ctk.CTkButton(
            mf, text="📥 Загрузить / Обучить модель",
            command=self._load_or_train_model, height=36)
        self.load_train_btn.pack(fill="x", padx=10, pady=(0, 10))

        ctk.CTkLabel(left, text="Активное обучение",
                     font=ctk.CTkFont(size=14, weight="bold")).pack(
            anchor="w", padx=8, pady=(12, 4))
        al = ctk.CTkFrame(left, corner_radius=8)
        al.pack(fill="x", padx=4, pady=4)

        self.ask_uncertain_var = tk.BooleanVar(value=True)
        ctk.CTkCheckBox(al, text="Спрашивать при неопределённости",
                        variable=self.ask_uncertain_var).pack(
            anchor="w", padx=10, pady=(8, 6))

        borders = ctk.CTkFrame(al, fg_color="transparent")
        borders.pack(fill="x", padx=10, pady=(0, 6))
        borders.grid_columnconfigure(0, weight=1)
        borders.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(borders, text="Нижняя граница, %", anchor="w").grid(
            row=0, column=0, sticky="w")
        ctk.CTkLabel(borders, text="Верхняя граница, %", anchor="w").grid(
            row=0, column=1, sticky="w", padx=(6, 0))
        self.low_var = tk.StringVar(value="40")
        ctk.CTkEntry(borders, textvariable=self.low_var).grid(
            row=1, column=0, sticky="ew", pady=(2, 0))
        self.high_var = tk.StringVar(value="60")
        ctk.CTkEntry(borders, textvariable=self.high_var).grid(
            row=1, column=1, sticky="ew", padx=(6, 0), pady=(2, 0))

        ctk.CTkLabel(al, text="Дообучать после N примеров:", anchor="w").pack(
            fill="x", padx=10, pady=(6, 2))
        self.min_samples_var = tk.StringVar(value="5")
        ctk.CTkEntry(al, textvariable=self.min_samples_var).pack(
            fill="x", padx=10, pady=(0, 6))

        ctk.CTkLabel(al, text="Эпох дообучения:", anchor="w").pack(
            fill="x", padx=10, pady=(0, 2))
        self.retrain_epochs_var = tk.StringVar(value="30")
        ctk.CTkEntry(al, textvariable=self.retrain_epochs_var).pack(
            fill="x", padx=10, pady=(0, 10))

        # ---------------- RIGHT COLUMN ------------------------------------
        right = ctk.CTkFrame(main, fg_color="transparent")
        right.grid(row=0, column=1, sticky="nsew")
        right.grid_rowconfigure(4, weight=1)
        right.grid_rowconfigure(6, weight=1)
        right.grid_columnconfigure(0, weight=1)

        btns = ctk.CTkFrame(right, corner_radius=8)
        btns.grid(row=0, column=0, sticky="ew", pady=(0, 8))

        self.start_btn = ctk.CTkButton(
            btns, text="▶  Запустить парсинг",
            command=self._start, height=40,
            fg_color="#3a7d3a", hover_color="#4a8d4a", width=200)
        self.start_btn.pack(side="left", padx=10, pady=10)

        self.stop_btn = ctk.CTkButton(
            btns, text="■  Остановить", command=self._stop, height=40,
            fg_color="#a03a3a", hover_color="#b04a4a",
            state="disabled", width=160)
        self.stop_btn.pack(side="left", padx=(0, 10), pady=10)

        ctk.CTkButton(
            btns, text="Очистить лог", command=self._clear_log, height=40,
            fg_color="#555555", hover_color="#666666", width=140
        ).pack(side="left", padx=(0, 10), pady=10)

        sub_header = ctk.CTkFrame(right, corner_radius=8)
        sub_header.grid(row=2, column=0, sticky="ew", pady=(0, 4))

        self.subs_count_label = ctk.CTkLabel(
            sub_header, text="Посылки (0)",
            font=ctk.CTkFont(size=14, weight="bold"))
        self.subs_count_label.pack(side="left", padx=10, pady=8)

        ctk.CTkButton(
            sub_header, text="🌐 Открыть в браузере",
            command=self._open_selected_submission, width=180,
            fg_color="#2b5a8a", hover_color="#3b6a9a", height=32
        ).pack(side="right", padx=6, pady=6)

        ctk.CTkButton(
            sub_header, text="📄 Показать код",
            command=self._show_selected_code, width=140,
            fg_color="#4a4a8a", hover_color="#5a5a9a", height=32
        ).pack(side="right", padx=6, pady=6)

        ctk.CTkButton(
            sub_header, text="🔁 Перепроверить",
            command=self._recheck_selected, width=140,
            fg_color="#7a5a2a", hover_color="#8a6a3a", height=32
        ).pack(side="right", padx=6, pady=6)

        ctk.CTkButton(
            sub_header, text="📋 Копировать ID",
            command=self._copy_selected_id, width=140,
            fg_color="#555555", hover_color="#666666", height=32
        ).pack(side="right", padx=6, pady=6)

        ctk.CTkButton(
            sub_header, text="💾 Экспорт CSV",
            command=self._export_csv, width=140,
            fg_color="#555555", hover_color="#666666", height=32
        ).pack(side="right", padx=6, pady=6)

        filter_frame = ctk.CTkFrame(right, corner_radius=8)
        filter_frame.grid(row=3, column=0, sticky="ew", pady=(0, 4))
        filter_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(filter_frame, text="Фильтр по вероятности ≥").grid(
            row=0, column=0, padx=(10, 6), pady=8, sticky="w")
        self.filter_var = tk.DoubleVar(value=0.0)
        self.filter_slider = ctk.CTkSlider(
            filter_frame, from_=0, to=100, number_of_steps=100,
            variable=self.filter_var, command=self._on_filter_change)
        self.filter_slider.grid(row=0, column=1, padx=6, pady=8, sticky="ew")
        self.filter_label = ctk.CTkLabel(filter_frame, text="0%", width=50)
        self.filter_label.grid(row=0, column=2, padx=(6, 10), pady=8)

        header_row = ctk.CTkFrame(right, corner_radius=4)
        header_row.grid(row=4, column=0, sticky="ew", pady=(0, 2))
        header_row.grid_columnconfigure(0, weight=0, minsize=110)
        header_row.grid_columnconfigure(1, weight=0, minsize=60)
        header_row.grid_columnconfigure(2, weight=0, minsize=140)
        header_row.grid_columnconfigure(3, weight=1, minsize=200)
        header_row.grid_columnconfigure(4, weight=0, minsize=140)
        header_row.grid_columnconfigure(5, weight=0, minsize=140)
        header_row.grid_columnconfigure(6, weight=0, minsize=100)

        for col, name in enumerate(
            ["ID", "Время", "Автор", "Задача", "Язык", "Вердикт", "Вероятность"]
        ):
            ctk.CTkLabel(
                header_row, text=name,
                font=ctk.CTkFont(size=12, weight="bold"),
                anchor="w"
            ).grid(row=0, column=col, sticky="ew", padx=6, pady=6)

        self.table_frame = ctk.CTkScrollableFrame(right)
        self.table_frame.grid(row=5, column=0, sticky="nsew", pady=(0, 4))
        self.table_frame.grid_columnconfigure(0, weight=0, minsize=110)
        self.table_frame.grid_columnconfigure(1, weight=0, minsize=60)
        self.table_frame.grid_columnconfigure(2, weight=0, minsize=140)
        self.table_frame.grid_columnconfigure(3, weight=1, minsize=200)
        self.table_frame.grid_columnconfigure(4, weight=0, minsize=140)
        self.table_frame.grid_columnconfigure(5, weight=0, minsize=140)
        self.table_frame.grid_columnconfigure(6, weight=0, minsize=100)

        # Лог
        log_frame = ctk.CTkFrame(right, corner_radius=8)
        log_frame.grid(row=6, column=0, sticky="nsew")
        log_frame.grid_rowconfigure(1, weight=1)
        log_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            log_frame, text="Лог",
            font=ctk.CTkFont(size=14, weight="bold"), anchor="w"
        ).grid(row=0, column=0, sticky="w", padx=10, pady=(8, 2))

        self.log_widget = ctk.CTkTextbox(
            log_frame, wrap="word", height=180,
            font=ctk.CTkFont(family="Consolas", size=12),
        )
        self.log_widget.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        self.log_widget.configure(state="disabled")

    # ---------------------------------------------------------------- helpers
    def _choose_model_dir(self):
        path = filedialog.askdirectory(title="Выберите папку с моделью")
        if path:
            self.model_path_var.set(path)

    def _choose_dataset(self):
        path = filedialog.askopenfilename(
            title="Выберите CSV-датасет",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if path:
            self.dataset_path_var.set(path)

    def _clear_log(self):
        self.log_widget.configure(state="normal")
        self.log_widget.delete("1.0", "end")
        self.log_widget.configure(state="disabled")

    def log(self, msg: str):
        def _append():
            self.log_widget.configure(state="normal")
            self.log_widget.insert("end", msg + "\n")
            self.log_widget.see("end")
            self.log_widget.configure(state="disabled")
        self.after(0, _append)

    # ---------------------------------------------------------- probability
    @staticmethod
    def _prob_color(prob):
        if prob is None:
            return "#666666"
        if prob > 60:
            return "#e06c75"
        if prob >= 40:
            return "#e5c07b"
        return "#98c379"

    # --------------------------------------------------- управление моделью
    def _load_or_train_model(self):
        if self.classifier is not None:
            if not messagebox.askyesno(
                "Модель уже загружена",
                "Модель уже загружена. Перезагрузить / переобучить её?"
            ):
                return

        model_path = self.model_path_var.get().strip()
        model_type = self.model_type_var.get().strip()

        if not model_path:
            messagebox.showwarning("Внимание", "Укажите путь к модели.")
            return

        if my_ai.model_exists(model_path=model_path, model_type=model_type):
            self.log(f"📂 Найдена существующая модель: {model_path} ({model_type})")
            self._load_model(model_path, model_type)
        else:
            self.log(f"❓ Модель не найдена по пути '{model_path}' ({model_type}).")
            ans = messagebox.askyesno(
                "Модель не найдена",
                f"Модель по пути '{model_path}' ({model_type}) не найдена.\n"
                f"Обучить новую на датасете "
                f"'{self.dataset_path_var.get()}'?"
            )
            if ans:
                self._train_model()

    def _load_model(self, model_path, model_type):
        try:
            self.log("Загрузка модели...")
            self.classifier = my_ai.load_model(
                model_path=model_path, model_type=model_type,
            )
            self.log("✅ Модель успешно загружена.")
        except Exception as e:
            self.log(f"❌ Ошибка загрузки модели: {e}")
            messagebox.showerror("Ошибка", f"Не удалось загрузить модель:\n{e}")

    def _train_model(self):
        data_path = self.dataset_path_var.get().strip()
        model_path = self.model_path_var.get().strip()
        model_type = self.model_type_var.get().strip()

        if not os.path.exists(data_path):
            messagebox.showerror("Ошибка", f"Датасет не найден:\n{data_path}")
            return

        try:
            epochs = int(self.train_epochs_var.get())
        except ValueError:
            messagebox.showerror("Ошибка", "Эпох обучения должно быть числом.")
            return

        self.load_train_btn.configure(state="disabled")
        self.start_btn.configure(state="disabled")

        self.train_thread = threading.Thread(
            target=self._train_worker,
            args=(data_path, model_path, model_type, epochs),
            daemon=True,
        )
        self.train_thread.start()

    def _train_worker(self, data_path, model_path, model_type, epochs):
        try:
            self.log(f"🚀 Начинаю обучение модели ({model_type}) "
                     f"на {data_path} ({epochs} эпох)...")
            classifier = my_ai.create_model(
                data_path=data_path, model_type=model_type,
                model_save_path=model_path, epochs=epochs,
                log_callback=self.log,
            )
            self.classifier = classifier
            self.log("✅ Модель обучена и готова к работе.")
        except Exception as e:
            self.log(f"🔥 Ошибка обучения модели: {e}")
            messagebox.showerror("Ошибка обучения", str(e))
        finally:
            self.after(0, self._on_training_done)

    def _on_training_done(self):
        self.load_train_btn.configure(state="normal")
        self.start_btn.configure(state="normal")

    # ---------------------------------------------------------------- запуск
    def _start(self):
        if self.classifier is None:
            messagebox.showwarning("Внимание",
                                   "Сначала загрузите или обучите модель.")
            return

        url = self.url_var.get().strip()
        if not url:
            messagebox.showwarning("Внимание", "Укажите ссылку или contestId.")
            return

        self._clear_table()

        self.active_learner = None
        if self.ask_uncertain_var.get():
            try:
                low = float(self.low_var.get())
                high = float(self.high_var.get())
                min_samples = int(self.min_samples_var.get())
                epochs = int(self.retrain_epochs_var.get())
            except ValueError:
                messagebox.showerror(
                    "Ошибка",
                    "Проверьте числовые параметры активного обучения.")
                return

            self.active_learner = ActiveLearner(
                classifier=self.classifier,
                dataset_path=self.dataset_path_var.get(),
                uncertain_low=low, uncertain_high=high,
                min_samples_to_retrain=min_samples,
                retrain_epochs=epochs, log_callback=self.log,
            )
            self.log(f"🎓 Активное обучение включено "
                     f"(серая зона: {low}–{high}%, порог: {min_samples})")

        self.stop_flag = False
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")

        self.parser_thread = threading.Thread(
            target=self._run_parser, args=(url,), daemon=True,
        )
        self.parser_thread.start()

    def _stop(self):
        self.stop_flag = True
        self.log("Запрошена остановка...")

    def _run_parser(self, url):
        try:
            parser_module.parse(
                url=url,
                classifier=self.classifier,
                log_callback=self.log,
                stop_check=lambda: self.stop_flag,
                save_dir="solutions",
                active_learner=self.active_learner,
                ask_callback=self._ask_user_callback,
                on_submission_found=self._on_submission_found,
            )
        except Exception as e:
            self.log(f"🔥 Критическая ошибка: {e}")
        finally:
            self.after(0, self._on_parser_done)

    def _on_parser_done(self):
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.log("Готово.")

    # -------------------------------------------------------- таблица
    def _clear_table(self):
        for widget in self.table_frame.winfo_children():
            widget.destroy()
        self.submissions.clear()
        self.id_to_row.clear()
        self.row_to_id.clear()
        self.selected_id = None
        self.subs_count_label.configure(text="Посылки (0)")

    def _on_submission_found(self, sub: dict):
        def _update():
            sid = str(sub["id"])

            if sid not in self.id_to_row:
                self._add_row(sub)
            else:
                self.submissions[sid].update(sub)
                self._refresh_row(sid)

        self.after(0, _update)

    def _add_row(self, sub: dict):
        sid = str(sub["id"])
        self.submissions[sid] = dict(sub)
        row_idx = len(self.submissions)

        row = ctk.CTkFrame(
            self.table_frame, corner_radius=4,
            fg_color="#2a2a2a" if row_idx % 2 == 0 else "#242424"
        )
        row.grid(row=row_idx - 1, column=0, columnspan=7, sticky="ew", pady=1)
        row.grid_columnconfigure(0, weight=0, minsize=110)
        row.grid_columnconfigure(1, weight=0, minsize=60)
        row.grid_columnconfigure(2, weight=0, minsize=140)
        row.grid_columnconfigure(3, weight=1, minsize=200)
        row.grid_columnconfigure(4, weight=0, minsize=140)
        row.grid_columnconfigure(5, weight=0, minsize=140)
        row.grid_columnconfigure(6, weight=0, minsize=100)

        values = [
            sid,
            sub.get("time", ""),
            sub.get("author", ""),
            sub.get("problem", ""),
            sub.get("lang", ""),
            sub.get("verdict", ""),
        ]
        labels = []
        for col, val in enumerate(values):
            lbl = ctk.CTkLabel(
                row, text=val, anchor="w",
                font=ctk.CTkFont(size=11)
            )
            lbl.grid(row=0, column=col, sticky="ew", padx=6, pady=4)
            labels.append(lbl)

        prob = sub.get("probability")
        prob_text = "—" if prob is None else f"{prob}%"
        prob_lbl = ctk.CTkLabel(
            row, text=prob_text, anchor="w",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=self._prob_color(prob),
        )
        prob_lbl.grid(row=0, column=6, sticky="ew", padx=6, pady=4)
        labels.append(prob_lbl)

        def on_click(_e, sid=sid):
            self._select(sid)

        def on_double(_e, sid=sid):
            self._open_submission(sid)

        for w in [row, *labels]:
            w.bind("<Button-1>", on_click)
            w.bind("<Double-Button-1>", on_double)

        self.id_to_row[sid] = row
        self.row_to_id[row] = sid
        self._refresh_counter()

    def _refresh_row(self, sid):
        sub = self.submissions[sid]
        row = self.id_to_row[sid]
        children = row.winfo_children()
        prob_lbl = children[6]
        prob = sub.get("probability")
        prob_lbl.configure(
            text="—" if prob is None else f"{prob}%",
            text_color=self._prob_color(prob),
        )

    def _refresh_counter(self):
        self.subs_count_label.configure(text=f"Посылки ({len(self.submissions)})")

    def _select(self, sid):
        if self.selected_id and self.selected_id in self.id_to_row:
            self.id_to_row[self.selected_id].configure(fg_color="#2a2a2a")
        self.selected_id = sid
        self.id_to_row[sid].configure(fg_color="#3a4a6a")

    def _selected(self):
        if not self.selected_id:
            return None
        return self.submissions.get(self.selected_id)

    # -------------------------------------------------- действия над строкой
    def _open_selected_submission(self):
        sub = self._selected()
        if not sub:
            messagebox.showinfo("Не выбрано", "Выберите посылку в таблице.")
            return
        self._open_submission(sub["id"])

    def _open_submission(self, sid):
        sub = self.submissions.get(sid)
        if not sub:
            return
        url = sub.get("url", "")
        if url:
            webbrowser.open(url)
            self.log(f"🌐 Открыт в браузере: {url}")
        else:
            self.log(f"⚠️ Нет URL для посылки {sid}")

    def _show_selected_code(self):
        sub = self._selected()
        if not sub:
            messagebox.showinfo("Не выбрано", "Выберите посылку в таблице.")
            return

        code = sub.get("code", "")
        if not code:
            saved = sub.get("saved_txt", "")
            if saved and os.path.exists(saved):
                try:
                    with open(saved, "r", encoding="utf-8") as f:
                        content = f.read()
                    # отрезаем заголовок
                    parts = content.split("\n\n", 1)
                    code = parts[1] if len(parts) > 1 else content
                except Exception as e:
                    self.log(f"⚠️ Не удалось прочитать файл: {e}")

        if not code:
            messagebox.showinfo(
                "Код недоступен",
                "Для этой посылки код ещё не загружен или пуст."
            )
            return

        CodeReviewDialog(
            self, code,
            prob=sub.get("probability") or 0.0,
            submission_data={
                "id": sub["id"], "time": sub.get("time", ""),
                "author": sub.get("author", ""), "problem": sub.get("problem", ""),
                "lang": sub.get("lang", ""), "verdict": sub.get("verdict", ""),
            }
        )

    def _copy_selected_id(self):
        sub = self._selected()
        if not sub:
            messagebox.showinfo("Не выбрано", "Выберите посылку в таблице.")
            return
        self.clipboard_clear()
        self.clipboard_append(str(sub["id"]))
        self.log(f"📋 ID скопирован: {sub['id']}")

    def _recheck_selected(self):
        sub = self._selected()
        if not sub:
            messagebox.showinfo("Не выбрано", "Выберите посылку в таблице.")
            return
        if self.classifier is None:
            messagebox.showwarning("Внимание", "Модель не загружена.")
            return

        code = sub.get("code", "")
        if not code:
            saved = sub.get("saved_txt", "")
            if saved and os.path.exists(saved):
                try:
                    with open(saved, "r", encoding="utf-8") as f:
                        content = f.read()
                    parts = content.split("\n\n", 1)
                    code = parts[1] if len(parts) > 1 else content
                except Exception:
                    pass

        if not code:
            messagebox.showinfo("Нет кода", "Для этой посылки код не загружен.")
            return

        try:
            ver = my_ai.check(self.classifier, code)
        except Exception as e:
            self.log(f"⚠️ Ошибка перепроверки: {e}")
            return

        sub["probability"] = ver
        self._refresh_row(str(sub["id"]))
        self.log(f"🔁 Перепроверено {sub['id']}: {ver}%")

    def _export_csv(self):
        if not self.submissions:
            messagebox.showinfo("Пусто", "Нет посылок для экспорта.")
            return
        path = filedialog.asksaveasfilename(
            title="Сохранить CSV",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")]
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8-sig", newline="") as f:
                w = csv.writer(f, delimiter=";")
                w.writerow(["ID", "Время", "Автор", "Задача",
                            "Язык", "Вердикт", "Вероятность", "URL"])
                for sid, sub in self.submissions.items():
                    w.writerow([
                        sid, sub.get("time", ""), sub.get("author", ""),
                        sub.get("problem", ""), sub.get("lang", ""),
                        sub.get("verdict", ""),
                        sub.get("probability", ""), sub.get("url", ""),
                    ])
            self.log(f"💾 Экспортировано в {path}")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить CSV:\n{e}")

    # -------------------------------------------------------- фильтр
    def _on_filter_change(self, value):
        self.filter_label.configure(text=f"{int(value)}%")
        self._apply_filter(value)

    def _apply_filter(self, min_prob):
        for sid, sub in self.submissions.items():
            row = self.id_to_row.get(sid)
            if not row:
                continue
            prob = sub.get("probability")
            if prob is None:
                row.grid()
                continue
            if prob >= min_prob:
                row.grid()
            else:
                row.grid_remove()

    # -------------------------------------------------- активное обучение
    def _ask_user_callback(self, code: str, prob: float, submission_data: dict):
        result_holder = {"label": None}
        done_event = threading.Event()

        def _ask():
            dialog = CodeReviewDialog(self, code, prob, submission_data)
            dialog.wait_window()
            result_holder["label"] = dialog.result
            done_event.set()

        self.after(0, _ask)
        done_event.wait()
        return result_holder["label"]


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
