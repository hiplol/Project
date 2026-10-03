# parser.py
import os
import csv
import time
import re

import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

import my_ai


PROFILE_DIR = os.path.abspath("chrome_profile")


# ------------------------------------------------------------------ helpers
def extract_contest_id(url: str):
    if url.isdigit():
        return int(url)
    m = re.search(r"/contest/(\d+)", url)
    return int(m.group(1)) if m else None


def extract_group_code(url: str):
    m = re.search(r"/group/([A-Za-z0-9]+)", url)
    return m.group(1) if m else None


def build_urls(contest_id: int, group_code: str = None):
    if group_code:
        base = f"https://codeforces.com/group/{group_code}/contest/{contest_id}"
    else:
        base = f"https://codeforces.com/contest/{contest_id}"
    return base, f"{base}/status"


def _make_driver(headless: bool = False):
    options = uc.ChromeOptions()
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--start-maximized")
    options.add_argument(f"--user-data-dir={PROFILE_DIR}")
    options.add_argument("--profile-directory=Default")
    options.add_argument("--disable-infobars")
    if headless:
        options.add_argument("--headless=new")
    driver = uc.Chrome(options=options, use_subprocess=True)
    driver.maximize_window()
    return driver


def _wait_cloudflare(driver, log, timeout: int = 60):
    log("⏳ Жду прохождения Cloudflare...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            title = driver.title.lower()
            page = driver.page_source.lower()
            if ("just a moment" not in title
                    and "checking your browser" not in page
                    and "attention required" not in title):
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def _is_logged_in(driver) -> bool:
    try:
        page = driver.page_source
        return "/logout" in page or "Выйти" in page or "Logout" in page
    except Exception:
        return False


def _ensure_logged_in(driver, url, log, wait_user_seconds: int = 300):
    log(f"🌐 Открываю: {url}")
    driver.get(url)
    _wait_cloudflare(driver, log, timeout=60)

    if _is_logged_in(driver):
        log("✅ Авторизация подтверждена (профиль сохранён).")
        return True

    log("⚠️ Похоже, ты не залогинен. У тебя есть "
        f"{wait_user_seconds} секунд, чтобы залогиниться в открытом браузере.")
    log("   После логина браузер продолжит работу автоматически.")

    start = time.time()
    while time.time() - start < wait_user_seconds:
        if _is_logged_in(driver):
            log("✅ Авторизация выполнена пользователем.")
            return True
        time.sleep(2)

    log("⌛ Время истекло, продолжаю без подтверждённого логина.")
    return False


def _safe_text(el):
    try:
        return el.text.strip()
    except Exception:
        return ""


def fetch_status_page(driver, status_url, log):
    log(f"🌐 Загружаю страницу посылок: {status_url}")
    driver.get(status_url)
    _wait_cloudflare(driver, log, timeout=60)
    time.sleep(2)

    possible_xpaths = [
        '//table[contains(@class, "status-frame-datatable")]/tbody',
        '//table[contains(@class, "status-frame")]/tbody',
        '//table[contains(@class, "status")]/tbody',
        '//*[@id="pageContent"]//table/tbody',
    ]

    table = None
    for xp in possible_xpaths:
        try:
            table = WebDriverWait(driver, 5).until(
                EC.presence_of_element_located((By.XPATH, xp))
            )
            log(f"✅ Найдена таблица посылок: {xp}")
            break
        except Exception:
            continue

    if table is None:
        log("❌ Не нашёл таблицу посылок на странице")
        with open("debug_status_page.html", "w", encoding="utf-8") as f:
            f.write(driver.page_source)
        log("   → сохранил HTML в debug_status_page.html")
        return []

    rows = table.find_elements(By.XPATH, './/tr[position()>1]')
    subs = []
    for row in rows:
        try:
            cells = row.find_elements(By.XPATH, './/td')
            if len(cells) < 6:
                continue
            link = row.find_element(
                By.XPATH, './/a[contains(@href, "/submission/")]'
            )
            sub_url = link.get_attribute("href") or ""

            subs.append({
                "id": _safe_text(cells[0]),
                "time": _safe_text(cells[1]),
                "author": _safe_text(cells[2]),
                "problem": _safe_text(cells[3]),
                "lang": _safe_text(cells[4]),
                "verdict": _safe_text(cells[5]).upper(),
                "url": sub_url,
            })
        except Exception:
            continue

    log(f"✅ Найдено {len(subs)} посылок на странице")
    return subs


def fetch_submission_code(driver, sub_url, log):
    driver.get(sub_url)
    _wait_cloudflare(driver, log, timeout=30)
    time.sleep(1)

    try:
        pre = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located(
                (By.XPATH, '//pre[@id="program-source-text"]')
            )
        )
        return pre.text or ""
    except Exception:
        try:
            pre = driver.find_element(
                By.XPATH, '//pre[@id="program-source-text"]'
            )
            return pre.text or ""
        except Exception:
            return ""


# ------------------------------------------------------------------ main
def parse(url: str,
          login: str = "",
          password: str = "",
          classifier=None,
          log_callback=print,
          stop_check=lambda: False,
          save_dir: str = "solutions",
          active_learner=None,
          ask_callback=None,
          choose_contests_callback=None,
          headless: bool = False,
          api_key: str = None,
          api_secret: str = None,
          cookie_string: str = None,
          on_submission_found=None):
    log = log_callback
    os.makedirs(save_dir, exist_ok=True)
    html_dir = os.path.join(save_dir, "html")
    os.makedirs(html_dir, exist_ok=True)

    contest_id = extract_contest_id(url.strip())
    if contest_id is None:
        log(f"❌ Не удалось извлечь contestId из ссылки: {url}")
        return

    group_code = extract_group_code(url)
    log(f"✅ contestId = {contest_id}"
        + (f", groupCode = {group_code}" if group_code else ""))
    log(f"📁 Профиль Chrome: {PROFILE_DIR}")

    driver = _make_driver(headless=headless)

    try:
        base_url, status_url = build_urls(contest_id, group_code)
        _ensure_logged_in(driver, "https://codeforces.com/",
                          log, wait_user_seconds=300)

        subs = fetch_status_page(driver, status_url, log)
        if not subs:
            log("❌ Список посылок пуст.")
            return

        rows = []
        for idx, sub in enumerate(subs, 1):
            if stop_check():
                log("⏹ Остановка по запросу пользователя.")
                break

            log(f"[{idx}/{len(subs)}] ID:{sub['id']} | "
                f"{sub['problem']} | {sub['lang']} | {sub['verdict']}")

            if on_submission_found:
                on_submission_found({**sub, "probability": None, "saved_txt": ""})

            try:
                code = fetch_submission_code(driver, sub["url"], log)
            except Exception as e:
                log(f"⚠️ Ошибка загрузки посылки {sub['id']}: {e}")
                continue

            if not code:
                log(f"   ⚠️ Пустой исходник для {sub['id']}, пропускаю.")
                continue

            try:
                with open(os.path.join(html_dir, f"{sub['id']}.html"),
                          "w", encoding="utf-8") as f:
                    f.write(driver.page_source)
            except Exception:
                pass

            ver = None
            clf_err = None
            if classifier is not None:
                try:
                    ver, clf_err = my_ai.check(classifier, code)
                except Exception as e:
                    clf_err = str(e)

            if ver is None:
                log(f"   ⚠️ Классификатор не смог обработать: {clf_err}")
            else:
                log(f"   → Вероятность списывания: {ver}%")

            if (active_learner is not None
                    and ver is not None
                    and active_learner.is_uncertain(ver)):
                log(f"🤔 Неопределённый случай ({ver}%). Спрашиваю пользователя...")
                if ask_callback is not None:
                    submission_data = {
                        "id": sub["id"],
                        "time": sub["time"],
                        "author": sub["author"],
                        "problem": sub["problem"],
                        "lang": sub["lang"],
                        "verdict": sub["verdict"],
                    }
                    label = ask_callback(code, ver, submission_data)
                    if label in (0, 1):
                        active_learner.add_example(code, label)
                        log(f"📝 Получена метка: {label}")
                    else:
                        log("⏭ Пользователь пропустил пример.")

            ver_str = "—" if ver is None else f"{ver}%"

            problem_name = sub["problem"].replace(" ", "_").replace("/", "_")
            filename = os.path.join(save_dir, f"{problem_name}_{sub['id']}.txt")
            with open(filename, "w", encoding="utf-8") as f:
                f.write(f"ID: {sub['id']}\n")
                f.write(f"Задача: {sub['problem']}\n")
                f.write(f"Автор: {sub['author']}\n")
                f.write(f"Время: {sub['time']}\n")
                f.write(f"Язык: {sub['lang']}\n")
                f.write(f"Вердикт: {sub['verdict']}\n")
                f.write(f"Вероятность списывания: {ver_str}\n\n")
                f.write(code)

            if on_submission_found:
                on_submission_found({
                    **sub,
                    "probability": ver,
                    "saved_txt": filename,
                    "code": code,
                })

            rows.append({
                "ID": sub["id"],
                "Время": sub["time"],
                "Автор": sub["author"],
                "Задача": sub["problem"],
                "Язык": sub["lang"],
                "Вердикт": sub["verdict"],
                "Вероятность": ver_str,
            })

            time.sleep(0.5)

        if rows:
            csv_path = os.path.join(save_dir, "results.csv")
            with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
                w = csv.DictWriter(
                    f, delimiter=";",
                    fieldnames=["ID", "Время", "Автор", "Задача",
                                "Язык", "Вердикт", "Вероятность"]
                )
                w.writeheader()
                w.writerows(rows)
            log(f"💾 Сводный CSV сохранён: {csv_path}")

    finally:
        try:
            driver.quit()
        except Exception:
            pass
        log("🔴 Парсер завершён.")
