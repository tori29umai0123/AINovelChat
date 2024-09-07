import csv
import datetime
import os
from typing import List


class Logger:
    LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")

    def list_log_files() -> List[str]:
        """ログファイルのリストを取得します"""
        if not os.path.exists(Logger.LOG_DIR):
            return []
        return [f for f in os.listdir(Logger.LOG_DIR) if f.endswith('.csv')]

    def load_chat_log(file_name: str) -> List[List[str]]:
        """チャットログを読み込みます"""
        file_path = os.path.join(Logger.LOG_DIR, file_name)
        chat_history = []
        with open(file_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # ヘッダーをスキップ
            for row in reader:
                if len(row) != 2:
                    continue
                role, message = row
                if role == "user":
                    chat_history.append([message, None])
                elif role == "assistant":
                    if chat_history and chat_history[-1][1] is None:
                        chat_history[-1][1] = message
                    else:
                        chat_history.append([None, message])
        return chat_history

    def save_chat_log(chat_history: List[List[str]]) -> str:
        """チャットログを保存します"""
        current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"{current_time}.csv"
        if not os.path.exists(Logger.LOG_DIR):
            os.makedirs(Logger.LOG_DIR)
        file_path = os.path.join(Logger.LOG_DIR, filename)
        
        with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Role", "Message"])
            for user_message, assistant_message in chat_history:
                if user_message:
                    writer.writerow(["user", user_message])
                if assistant_message:
                    writer.writerow(["assistant", assistant_message])
        
        return f"チャットログが {file_path} に保存されました。"