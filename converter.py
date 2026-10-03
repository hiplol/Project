# converter.py
import pyperclip

a = """import sys
input = sys.stdin.read().split()
ptr = 0
n = int(input[ptr])
ptr += 1
events = []
for _ in range(n):
    a = int(input[ptr])
    b = int(input[ptr + 1])
    ptr += 2
    events.append((a, 1))
    events.append((b, -1))  

events.sort(key=lambda x: (x[0], x[1]))
max_friends = 0
current_friends = 0
best_time = None
for time, delta in events:
    current_friends += delta
    if current_friends > max_friends:
        max_friends = current_friends
        best_time = time
print(max_friends, best_time)
"""

pyperclip.copy('\\n '.join(a.split('\n')))
