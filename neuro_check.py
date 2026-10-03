# neuro_check.py
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

texts = [
    "def solve(x): return x * 2",
    "def solve(x): return x + x",
    "def calculate(y): return y * 2",
    "def solve(a): return a * 2",
    "def sl(m) return m * 5",
    "def to_lower(string) return string.lower()"
]
labels = np.array([0, 1, 0, 1, 0, 1])

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts).toarray()

X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, random_state=42
)

model = Sequential([
    Dense(64, activation='relu', input_shape=(X.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=200, batch_size=2, validation_split=0.1)

loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nТочность модели: {accuracy * 100:.2f}%")


def predict_plagiarism(code):
    vec = vectorizer.transform([code]).toarray()
    prob = model.predict(vec)[0][0]
    return prob * 100
