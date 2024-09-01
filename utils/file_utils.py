import os
from typing import List

class FileUtils:
    @staticmethod
    def get_files_with_extension(directory: str, extension: str) -> List[str]:
        """指定されたディレクトリから特定の拡張子を持つファイルのリストを取得します"""
        return [f for f in os.listdir(directory) if f.endswith(extension)]