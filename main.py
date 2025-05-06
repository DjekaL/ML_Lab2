import os
import io
import csv
import json
import numpy as np
from tqdm import tqdm
import ffmpeg
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
import torch
from transformers import pipeline, AutoTokenizer, AutoModel
import time
import subprocess



# Проверка GPU
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('medium')

# Инициализация моделей
asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-small",
    device=0 if torch.cuda.is_available() else -1,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
embedding_model = AutoModel.from_pretrained(embedding_model_name).to(device)
if torch.cuda.is_available():
    embedding_model = embedding_model.half()

# Функции обработки
def extract_video_id(url):
    # Обрабатываем как полные URL, так и короткие ID
    if 'youtube.com/watch?v=' in url:
        return url.split('v=')[1].split('&')[0]
    elif 'youtu.be/' in url:
        return url.split('youtu.be/')[1].split('?')[0]
    elif len(url.strip()) == 11:  # Если передан чистый ID
        return url.strip()
    return None  # Если ID не распознан

def get_links_with_labels(csv_path):
    links = []
    labels = []
    category_map = {}
    
    with open(csv_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file, delimiter=',')  # Исправлено на запятую
        for row in reader:
            url = row.get('link', '').strip()  # Колонка 'link' вместо 'URL'
            category = row.get('category', '').strip().lower()  # Колонка 'category'
            
            if not url or not category:
                continue
                
            video_id = extract_video_id(url)
            if video_id:
                if category not in category_map:
                    category_map[category] = len(category_map)
                
                links.append(f"https://www.youtube.com/watch?v={video_id}")
                labels.append(category_map[category])
    
    return links, labels, category_map

def download_to_memory(url):
    #time.sleep(random.uniform(3, 5))
    output_path = "temp_video.mp4"
    try:
        subprocess.run([
            "yt-dlp",
            "--cookies-from-browser", "firefox",
            "-f", "worst[ext=mp4]",
            "-o", output_path,
            url
        ], check=True)
        with open(output_path, "rb") as f:
            buffer = io.BytesIO(f.read())
        os.remove(output_path)
        buffer.seek(0)
        return buffer
    except Exception as e:
        print(f"yt-dlp download error: {str(e)}")
        return None

def extract_audio_from_memory(video_buffer):
    print("extract_audio_from_memory")
    try:
        input_data = video_buffer.getvalue()
        process = (
            ffmpeg
            .input('pipe:', format='mp4')
            .output('pipe:', format='wav', acodec='pcm_s16le', ar='16000', ac=1)
            .run_async(pipe_stdin=True, pipe_stdout=True, quiet=True)
        )
        out, _ = process.communicate(input=input_data)
        return io.BytesIO(out)
    except Exception as e:
        print(f"Error extracting audio: {str(e)}")
        return None

def transcribe_audio(audio_buffer):
    print("transcribe_audio")
    try:
        audio_data = np.frombuffer(audio_buffer.getvalue(), dtype=np.int16)
        audio_data = audio_data.astype(np.float32) / 32768.0
        
        chunk_size = 30 * 16000  # 30 секунд при 16кГц
        text_parts = []
        
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i+chunk_size]
            result = asr_pipeline(
                {"raw": chunk, "sampling_rate": 16000},
                return_timestamps=False,
                generate_kwargs={
                    "language": "english",
                    "temperature": 0.0,  # Для более детерминированного вывода
                    #"no_speech_threshold": 0.6,  # Игнорировать фрагменты без речи
                    "compression_ratio_threshold": 2.4  # Фильтр бессмысленных повторов
                }
            )
            text_parts.append(result["text"])
        
        return " ".join(text_parts)
    except Exception as e:
        print(f"Ошибка транскрипции: {str(e)}")
        return ""

def get_text_embeddings(text):
    print("get_text_embeddings")
    try:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = embedding_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]
    except Exception as e:
        print(f"Embedding error: {str(e)}")
        return np.zeros(768)

def save_texts_with_labels(texts, labels, valid_indices, category_map, filename="texts_with_labels.json"):
    """Сохраняет тексты с соответствующими метками в JSON файл"""
    data = []
    reverse_category_map = {v: k for k, v in category_map.items()}
    
    for idx, text in zip(valid_indices, texts):
        label_id = labels[idx]
        data.append({
            "text": text,
            "label_id": label_id,
            "label_name": reverse_category_map.get(label_id, "unknown")
        })
    
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"Saved texts with labels to {filename}")

def load_texts_with_labels(filename="texts_with_labels.json"):
    """Загружает тексты с метками из файла"""
    try:
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        texts = [item["text"] for item in data]
        labels = [item["label_id"] for item in data]
        
        # Восстанавливаем category_map
        category_map = {}
        for item in data:
            label_name = item["label_name"]
            label_id = item["label_id"]
            if label_name not in category_map:
                category_map[label_name] = label_id
        
        return texts, labels, category_map
    except Exception as e:
        print(f"Error loading texts with labels: {str(e)}")
        return None, None, None

def process_from_file(filename="texts_with_labels.json"):
    """Обрабатывает данные из сохраненного файла с текстами"""
    texts, labels, category_map = load_texts_with_labels(filename)
    if texts is None:
        return None
    
    print(f"Loaded {len(texts)} texts from file")
    
    # Получаем эмбеддинги для текстов
    embeddings = []
    for text in tqdm(texts, desc="Processing texts"):
        embedding = get_text_embeddings(text)
        embeddings.append(embedding)
    
    if len(embeddings) < 2:
        print("Not enough samples for training")
        return None
    
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, labels, test_size=0.2, random_state=42
    )
    
    print("\n=== Распределение меток ===")
    print(f"Всего образцов: {len(labels)}")
    print(f"Обучающая выборка: {len(y_train)} образцов")
    print(f"Тестовая выборка: {len(y_test)} образцов")
    
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    print(f"Model accuracy: {accuracy:.2f}")
    
    if len(X_train) > 1:
        n_splits = min(3, len(X_train))
        scores = cross_val_score(clf, X_train, y_train, cv=n_splits)
        print(f"Cross-validation scores: {scores}")
        print(f"Mean CV accuracy: {scores.mean():.2f} (±{scores.std():.2f})")
    
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    return clf, np.array(embeddings), texts, list(range(len(texts))), category_map

def process_videos(links, labels=None):
    embeddings = []
    texts = []
    valid_indices = []
    
    for i, link in enumerate(tqdm(links, desc="Processing videos")):
        video_buffer = download_to_memory(link)
        if not video_buffer:
            continue
            
        audio_buffer = extract_audio_from_memory(video_buffer)
        if not audio_buffer:
            continue
            
        text = transcribe_audio(audio_buffer)
        if not text:
            continue
            
        embedding = get_text_embeddings(text)
        
        embeddings.append(embedding)
        texts.append(text)
        valid_indices.append(i)
    
    if labels is None:
        return np.array(embeddings), texts
    
    # Сохраняем тексты с метками
    save_texts_with_labels(texts, labels, valid_indices, category_map)
    
    if len(embeddings) < 2:  # Нужно хотя бы 2 образца для split
        print(f"Warning: Only {len(embeddings)} samples processed. Need at least 2.")
        return None

    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, [labels[i] for i in valid_indices], test_size=0.2, random_state=42
    )

    print("\n=== Распределение меток ===")
    print(f"Всего образцов: {len(valid_indices)}")
    print(f"Обучающая выборка: {len(y_train)} образцов")
    print(f"Тестовая выборка: {len(y_test)} образцов")
    
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    print(f"Model accuracy: {accuracy:.2f}")
    
    if len(X_train) > 1:
        n_splits = min(3, len(X_train))
        scores = cross_val_score(clf, X_train, y_train, cv=n_splits)
        print(f"Cross-validation scores: {scores}")
        print(f"Mean CV accuracy: {scores.mean():.2f} (±{scores.std():.2f})")
    
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    return clf, np.array(embeddings), texts, valid_indices, category_map

# Основной код
if __name__ == "__main__":
    # 1. Загрузка данных
    csv_path = "youtube.csv"
    print(f"Loading data from {csv_path}...")
    video_links, labels, category_map = get_links_with_labels(csv_path)
    
    print("\n=== Сопоставление меток и категорий ===")
    for category, label_id in category_map.items():
        print(f"Метка {label_id}: {category}")

    if not video_links:
        print("Error: No valid links found in CSV.")
        exit()

    # 2. Выбор режима работы
    print("\nВыберите режим работы:")
    print("1 - Обработать видео и создать новый набор данных")
    print("2 - Загрузить существующие тексты и метки из файла")
    choice = input("Ваш выбор (1/2): ").strip()
    
    if choice == "1":
        # Обработка видео
        print(f"Processing {len(video_links)} videos...")
        results = process_videos(video_links, labels)  # Первые 1000 для примера
    elif choice == "2":
        # Загрузка из файла
        results = process_from_file()
    else:
        print("Неверный выбор. Завершение работы.")
        exit()

    if results is None:
        print("Processing failed. Check errors above.")
        exit()
    
    # 3. Сохранение результатов
    if len(results) == 5:
        classifier, embeddings, texts, indices, category_map = results
        torch.save(classifier, "classifier.pth")
        np.save("embeddings.npy", embeddings)
        
        with open("category_map.json", "w") as f:
            json.dump(category_map, f)
        
        print("Processing completed. Saved:")
        print("- classifier.pth")
        print("- embeddings.npy")
        print("- category_map.json")
        print("- texts_with_labels.json (created automatically)")