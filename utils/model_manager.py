from typing import List, Tuple
from utils.file_utils import FileUtils

class ModelManager:
    def __init__(self, model_dir: str, model_extension: str = '.gguf'):
        self.model_dir = model_dir
        self.model_extension = model_extension

    @staticmethod
    def get_model_files(model_dir: str, model_extension: str = '.gguf') -> List[str]:
        """モデルファイルのリストを取得します"""
        return FileUtils.get_files_with_extension(model_dir, model_extension)

    def update_model_dropdown(self, current_value: str) -> Tuple[List[str], str, str]:
        """モデルドロップダウンを更新します"""
        model_files = self.get_model_files(self.model_dir, self.model_extension)
        
        if current_value not in model_files:
            download_message = f"現在のモデル（{current_value}）が見つかりません。ダウンロードしてください。"
            model_files.insert(0, current_value)
        else:
            download_message = ""
        
        return model_files, current_value, download_message