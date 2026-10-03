from curl_cffi import requests
import re

COOKIE = "JSESSIONID=...; 39ce7=...; 70a7c28f3de=..."  # твоя строка
URL = "https://codeforces.com/group/bVSVAj3Smd/contest/643263/submission/348100087"

s = requests.Session(impersonate="chrome120")
s.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
})

for pair in COOKIE.split(";"):
    if "=" in pair:
        k, v = pair.strip().split("=", 1)
        s.cookies.set(k.strip(), v.strip(), domain=".codeforces.com")

r = s.get(URL)
print("Status:", r.status_code)
if r.status_code == 200:
    m = re.search(r'<pre id="program-source-text"[^>]*>(.*?)</pre>', r.text, re.DOTALL)
    print("Code found:", bool(m))
    if m:
        print(m.group(1)[:200])