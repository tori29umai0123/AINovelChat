import configparser
from typing import Any

class Config:
    def __init__(self, filename: str):
        self.filename = filename
        self.config = configparser.ConfigParser()
        self.load()

    def load(self) -> None:
        """設定ファイルを読み込みます"""
        self.config.read(self.filename, encoding='utf-8')

    def save(self) -> None:
        """設定をファイルに保存します"""
        with open(self.filename, 'w', encoding='utf-8') as configfile:
            self.config.write(configfile)

    def get(self, section: str, key: str, fallback: Any = None) -> Any:
        """指定されたセクションとキーの値を取得します"""
        return self.config.get(section, key, fallback=fallback)

    def set(self, section: str, key: str, value: Any) -> None:
        """指定されたセクションとキーに値を設定します"""
        if section not in self.config:
            self.config[section] = {}
        self.config[section][key] = str(value)

    def update(self, section: str, key: str, value: Any) -> str:
        """設定を更新し、ファイルに保存します"""
        self.set(section, key, value)
        self.save()
        return f"設定を更新しました: [{section}] {key} = {value}"

    def get_cohere_api_key(self) -> str:
        """Cohere APIキーを取得します"""
        return self.get('Cohere', 'api_key', fallback='')

    def set_cohere_api_key(self, api_key: str) -> None:
        """Cohere APIキーを設定し保存します"""
        self.set('Cohere', 'api_key', api_key)
        self.save()