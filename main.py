
import os
import requests
import base64
import json
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Конфигурация
OPENROUTER_API_KEY = "" #your api
OPENROUTER_MODEL = "nvidia/nemotron-nano-12b-v2-vl:free" #"arcee-ai/trinity-large-preview:free"  
GOOGLE_DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1x6EKNkVw6PlFVTr6cGrsVscmRuwqGrXd"
DOWNLOAD_DIR = "downloaded_files"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class FileData:
    """Класс для хранения данных о файле"""
    filepath: str
    filename: str
    mime_type: str
    base64_data: str

class FileDownloader:
    """Загрузчик файлов из Google Drive"""
    
    @staticmethod
    def extract_folder_id(url: str) -> str:
        if "folders/" in url:
            return url.split("folders/")[1].split("?")[0].split("/")[0]
        raise ValueError("Некорректный URL папки Google Drive")
    
    @staticmethod
    def get_direct_download_link(file_id: str) -> str:
        return f"https://drive.google.com/uc?export=download&id={file_id}"
    
    def download_from_folder(self, folder_url: str, download_dir: str = DOWNLOAD_DIR) -> List[str]:
        logger.info(f"Загрузка файлов из папки: {folder_url}")
        
        folder_id = self.extract_folder_id(folder_url)
        
        
        file_links = {
            "5_grazhdanskijj_kodeks_rossijjskojj_federacii_chast_pervaya.pdf": 
                "1nkfJ_rZBuJqRPBNkp8SzFUQ_NNtmwL1f",
            "3_fc21cff16e05519324936c9c89d642bb_1460720162.jpg": 
                "1jl-RdFVznqwcl2V4SGJ_d95SfxNvaFPj",
            "e6ce831e-6346-4397-b2a0-47e9470304a6.webp": 
                "1fuKnTaFWrBokILmh0I9xKXq4yBpSGB2A"
        }
        
        downloaded_files = []
        
        for filename, file_id in file_links.items():
            try:
                download_url = self.get_direct_download_link(file_id)
                filepath = os.path.join(download_dir, filename)
                
                logger.info(f"Скачивание {filename}...")
                response = requests.get(download_url, stream=True)
                response.raise_for_status()
                
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                downloaded_files.append(filepath)
                logger.info(f"Файл сохранен: {filepath}")
                
            except Exception as e:
                logger.error(f"Ошибка при загрузке {filename}: {e}")
        
        return downloaded_files

class FilePreparer:
    """Подготовка файлов для отправки в OpenRouter"""
    
    MIME_TYPES = {
        '.pdf': 'application/pdf',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.webp': 'image/webp',
        '.txt': 'text/plain',
        '.json': 'application/json',
    }
    
    @staticmethod
    def encode_file_to_base64(filepath: str) -> str:
        """Кодирует файл в base64"""
        try:
            with open(filepath, "rb") as f:
                encoded = base64.b64encode(f.read()).decode('utf-8')
            logger.info(f"Файл закодирован в base64: {filepath}")
            return encoded
        except Exception as e:
            logger.error(f"Ошибка при кодировании файла {filepath}: {e}")
            return ""
    
    @staticmethod
    def get_mime_type(filename: str) -> str:
        """Определяет MIME-тип по расширению файла"""
        ext = os.path.splitext(filename)[1].lower()
        return FilePreparer.MIME_TYPES.get(ext, 'application/octet-stream')
    
    def prepare_files(self, filepaths: List[str]) -> List[FileData]:
        """Подготавливает список файлов для отправки"""
        prepared_files = []
        
        for filepath in filepaths:
            if not os.path.exists(filepath):
                logger.warning(f"Файл не найден: {filepath}")
                continue
            
            filename = os.path.basename(filepath)
            mime_type = self.get_mime_type(filename)
            base64_data = self.encode_file_to_base64(filepath)
            
            if base64_data:
                prepared_files.append(FileData(
                    filepath=filepath,
                    filename=filename,
                    mime_type=mime_type,
                    base64_data=base64_data
                ))
                logger.info(f"Файл подготовлен: {filename} ({mime_type})")
        
        return prepared_files

class OpenRouterClient:
    """Клиент для работы с OpenRouter API с поддержкой файлов"""
    
    def __init__(self, api_key: str, model: str = OPENROUTER_MODEL):
        self.api_key = api_key
        self.model = model
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
    
    def create_file_message_content(self, file_data: FileData, prompt_text: str = "") -> List[Dict]:
        """Создает контент сообщения с файлом"""
        content_items = []
        
        if prompt_text:
            content_items.append({
                "type": "text",
                "text": prompt_text
            })
        
        if file_data.mime_type.startswith('image/'):
            # Для изображений
            content_items.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{file_data.mime_type};base64,{file_data.base64_data}"
                }
            })
        else:
            # Для документов (PDF, TXT и др.)
            content_items.append({
                "type": "file",
                "file": {
                    "filename": file_data.filename,
                    "file_data": f"data:{file_data.mime_type};base64,{file_data.base64_data}"
                }
            })
        
        return content_items
    
    def summarize_files(self, files_data: List[FileData]) -> Dict[str, Any]:
        """
        Отправляет файлы в OpenRouter и получает общее резюме.
        
        Args:
            files_data: Список подготовленных файлов
        
        Returns:
            Словарь с результатами
        """
        if not files_data:
            return {"error": "Нет файлов для обработки"}
        
        system_prompt = """Ты - помощник для анализа документов. 
        Проанализируй предоставленные файлы и создай единое, структурированное резюме.
        Учти информацию из всех источников, выдели ключевые моменты и основные темы."""
        
        user_prompt = """Проанализируй все предоставленные файлы и создай подробное общее резюме, которое включает:
        1. Основные темы и ключевые идеи из всех документов
        2. Важные детали и факты
        3. Общие выводы и наблюдения    
        Резюме должно быть на русском языке и полезным для понимания общего содержания всех файлов."""
        
        # Формируем сообщения для API
        messages = [
            {
                "role": "system",
                "content": system_prompt
            }
        ]
        
        # Добавляем каждый файл с пояснением
        for i, file_data in enumerate(files_data):
            file_prompt = f"Вот файл {i+1} из {len(files_data)}: {file_data.filename}"
            content = self.create_file_message_content(file_data, file_prompt)
            
            messages.append({
                "role": "user",
                "content": content
            })
        
        # Финальный промпт для суммаризации
        messages.append({
            "role": "user",
            "content": user_prompt
        })
        
        # Подготавливаем запрос
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://nobody.com",
            "X-Title": "Document Summarization Tool"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 4000,
            "temperature": 0.3
        }
        
        # Добавляем плагины для обработки файлов
        plugins = []
        
        # Для PDF файлов добавляем парсер
        if any(f.mime_type == 'application/pdf' for f in files_data):
            plugins.append({
                "id": "file-parser",
                "pdf": {"engine": "pdf-text"}  # Используем pdf-text для текстовых PDF
            })
        
        if plugins:
            payload["plugins"] = plugins
        
        try:
            logger.info(f"Отправка {len(files_data)} файлов в OpenRouter (модель: {self.model})...")
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            
            result = response.json()
            
            if "choices" in result and len(result["choices"]) > 0:
                summary = result["choices"][0]["message"]["content"]
                
                return {
                    "success": True,
                    "summary": summary,
                    "model": self.model,
                    "files_processed": len(files_data),
                    "file_names": [f.filename for f in files_data],
                    "raw_response": result
                }
            else:
                logger.error(f"Неожиданный формат ответа от OpenRouter: {result}")
                return {"error": "Неожиданный формат ответа от LLM"}
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Ошибка при запросе к OpenRouter: {e}")
            return {"error": f"Ошибка при обращении к LLM: {e}"}
        except Exception as e:
            logger.error(f"Неожиданная ошибка: {e}")
            return {"error": f"Неожиданная ошибка: {e}"}

class DocumentSummarizer:
    """Основной класс для управления процессом суммаризации"""
    
    def __init__(self, openrouter_api_key: str, openrouter_model: str = OPENROUTER_MODEL):
        self.downloader = FileDownloader()
        self.preparer = FilePreparer()
        self.llm_client = OpenRouterClient(openrouter_api_key, openrouter_model)
    
    def summarize_folder(self, folder_url: str, download_dir: str = DOWNLOAD_DIR) -> Dict[str, Any]:
        """
        Основной метод: скачивает файлы и отправляет их в OpenRouter для суммаризации.
        
        Возвращает словарь с результатами.
        """
        logger.info("Начало процесса суммаризации документов (прямая отправка файлов)...")
        
        # 1. Загрузка файлов
        downloaded_files = self.downloader.download_from_folder(folder_url, download_dir)
        
        if not downloaded_files:
            logger.error("Не удалось загрузить файлы. Завершение работы.")
            return {"error": "Не удалось загрузить файлы"}
        
        logger.info(f"Загружено {len(downloaded_files)} файлов")
        
        # 2. Подготовка файлов (кодирование в base64)
        prepared_files = self.preparer.prepare_files(downloaded_files)
        
        if not prepared_files:
            logger.error("Не удалось подготовить файлы для отправки.")
            return {"error": "Не удалось подготовить файлы для отправки"}
        
        logger.info(f"Подготовлено {len(prepared_files)} файлов для отправки в OpenRouter")
        
        # 3. Отправка файлов в OpenRouter и получение резюме
        logger.info("Отправка файлов в OpenRouter для анализа...")
        result = self.llm_client.summarize_files(prepared_files)
        
        # 4. Добавляем информацию о файлах в результат
        if "error" not in result:
            result["downloaded_files"] = downloaded_files
            result["prepared_files"] = [f.filename for f in prepared_files]
            result["stats"] = {
                "total_files": len(downloaded_files),
                "files_prepared": len(prepared_files),
                "summary_length": len(result.get("summary", ""))
            }
        
        logger.info("Процесс суммаризации завершен!")
        return result

# Вспомогательные функции
def save_results(results: Dict[str, Any], output_file: str = "summary_results_v2.json"):
    """Сохраняет результаты в JSON файл."""
    try:
        # Удаляем raw_response из сохранения, если он слишком большой
        save_results = results.copy()
        if "raw_response" in save_results:
            del save_results["raw_response"]
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(save_results, f, ensure_ascii=False, indent=2)
        logger.info(f"Результаты сохранены в файл: {output_file}")
        return True
    except Exception as e:
        logger.error(f"Ошибка при сохранении результатов: {e}")
        return False

def print_summary(results: Dict[str, Any]):
    """Выводит резюме и статистику в консоль."""
    if "error" in results:
        print(f"Ошибка: {results['error']}")
        return
    
    print("=" * 80)
    print("ОБЩЕЕ РЕЗЮМЕ ДОКУМЕНТОВ (отправка файлов напрямую)")
    print("=" * 80)
    print("\n" + results.get("summary", "Резюме недоступно"))
    
    if "stats" in results:
        print("\n" + "=" * 80)
        print("СТАТИСТИКА")
        print("=" * 80)
        stats = results["stats"]
        print(f"Всего файлов: {stats['total_files']}")
        print(f"Файлов обработано: {stats['files_prepared']}")
        print(f"Длина резюме: {stats['summary_length']} символов")
    
    if "file_names" in results:
        print("\n" + "=" * 80)
        print("ОБРАБОТАННЫЕ ФАЙЛЫ")
        print("=" * 80)
        for i, filename in enumerate(results["file_names"], 1):
            print(f"{i}. {filename}")

# Точка входа
if __name__ == "__main__":
    # Настройки
    API_KEY = OPENROUTER_API_KEY  
    
    if API_KEY == "your-api-key-here":
        print("ПРЕДУПРЕЖДЕНИЕ: Установите ваш API ключ OpenRouter в переменной OPENROUTER_API_KEY")
        print("Получите ключ на https://openrouter.ai")
        print("Запуск в демонстрационном режиме без реального обращения к API...")
        # Можно создать тестовый объект с заглушкой
        API_KEY = "demo-key-for-display"
    
    # Создаем суммаризатор и запускаем процесс
    summarizer = DocumentSummarizer(API_KEY)
    
    try:
        results = summarizer.summarize_folder(GOOGLE_DRIVE_FOLDER_URL)
        
        print_summary(results)
        
        if "error" not in results:
            save_results(results)
        else:
            print(f"\nОшибка при обработке: {results['error']}")
        
    except Exception as e:
        logger.error(f"Критическая ошибка в работе приложения: {e}")
        print(f"Произошла ошибка: {e}. Подробности в логах.")
