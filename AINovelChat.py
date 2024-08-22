import os
import sys
import time
import socket
import gradio as gr
from llama_cpp import Llama
import datetime
from jinja2 import Template
import configparser
from functools import partial
import threading
import asyncio
import csv
from typing import List, Dict, Any, Tuple

# 定数
DEFAULT_INI_FILE = 'settings.ini'
MODEL_FILE_EXTENSION = '.gguf'

# パスの設定
if getattr(sys, 'frozen', False):
    BASE_PATH = os.path.dirname(sys.executable)
    MODEL_DIR = os.path.join(os.path.dirname(BASE_PATH), "AINovelChat", "models")
else:
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))
    MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

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

class FileUtils:
    @staticmethod
    def get_files_with_extension(directory: str, extension: str) -> List[str]:
        """指定されたディレクトリから特定の拡張子を持つファイルのリストを取得します"""
        return [f for f in os.listdir(directory) if f.endswith(extension)]

class NetworkUtils:
    @staticmethod
    def get_ip_address() -> str:
        """ローカルIPアドレスを取得します"""
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            try:
                s.connect(('10.255.255.255', 1))
                return s.getsockname()[0]
            except Exception:
                return '127.0.0.1'

    @staticmethod
    def find_available_port(starting_port: int) -> int:
        """利用可能なポートを見つけます"""
        port = starting_port
        while NetworkUtils.is_port_in_use(port):
            print(f"ポート {port} は使用中です。次のポートを試します。")
            port += 1
        return port

    @staticmethod
    def is_port_in_use(port: int) -> bool:
        """指定されたポートが使用中かどうかを確認します"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) == 0

class ModelManager:
    def __init__(self, model_dir: str, model_extension: str = MODEL_FILE_EXTENSION):
        self.model_dir = model_dir
        self.model_extension = model_extension

    @staticmethod
    def get_model_files() -> List[str]:
        """モデルファイルのリストを取得します"""
        return FileUtils.get_files_with_extension(MODEL_DIR, MODEL_FILE_EXTENSION)

    def update_model_dropdown(self, current_value: str) -> Tuple[List[str], str, str]:
        """モデルドロップダウンを更新します"""
        model_files = self.get_model_files()
        
        if current_value not in model_files:
            download_message = f"現在のモデル（{current_value}）が見つかりません。ダウンロードしてください。"
            model_files.insert(0, current_value)
        else:
            download_message = ""
        
        return model_files, current_value, download_message

class LlamaAdapter:
    def __init__(self, model_path: str, params: Dict[str, Any]):
        self.model_path = model_path
        self.params = params
        self.llm = Llama(model_path=model_path, **params)

    def _process_response(self, response: Any) -> str:
        """LLMの応答を処理します"""
        if isinstance(response, dict) and "choices" in response:
            return response["choices"][0]["text"]
        elif isinstance(response, str):
            return response
        else:
            raise ValueError(f"予期しない応答形式です: {type(response)}")

    def generate_text(self, prompt: str, max_tokens: int) -> str:
        """テキストを生成します"""
        response = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=self.params['temperature'],
            top_p=self.params['top_p'],
            top_k=self.params['top_k'],
            repeat_penalty=self.params['repeat_penalty'],
            stop=["user:", "・会話履歴", "<END>"]
        )
        return self._process_response(response)

    def create_chat_completion(self, messages: List[Dict[str, str]], max_tokens: int) -> Dict[str, Any]:
        """チャット形式の応答を生成します"""
        return self.llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=self.params['temperature'],
            top_p=self.params['top_p'],
            top_k=self.params['top_k'],
            repeat_penalty=self.params['repeat_penalty']
        )

class CharacterMaker:
    def __init__(self, llm_adapter: LlamaAdapter, settings: Dict[str, Any]):
        self.llm_adapter = llm_adapter
        self.settings = settings
        self.history: List[Dict[str, str]] = []
        self.chat_history: List[Dict[str, str]] = []
        self.use_chat_format = False
        self.model_loaded = threading.Event()
        self.current_model = None
        self.model_lock = threading.Lock()
        self.chat_model_settings = None
        self.gen_model_settings = None

    def load_model(self, model_type: str) -> None:
        """モデルをロードします"""
        with self.model_lock:
            new_settings = self.get_current_settings(model_type)
            
            if model_type == 'CHAT':
                if self.chat_model_settings == new_settings:
                    print("CHATモデルの設定に変更がないため、リロードをスキップします。")
                    return
                self.chat_model_settings = new_settings
            else:  # GEN
                if self.gen_model_settings == new_settings:
                    print("GENモデルの設定に変更がないため、リロードをスキップします。")
                    return
                self.gen_model_settings = new_settings

            if self.are_models_identical():
                if self.llm_adapter and self.current_model == 'SHARED':
                    print("CHATモデルとGENモデルの設定が同じで、既にロードされています。リロードをスキップします。")
                    return
                print("CHATモデルとGENモデルの設定が同じです。共有モデルとしてロードします。")
                self.reload_model('SHARED', new_settings)
            else:
                print(f"{model_type}モデルをロードします。")
                self.reload_model(model_type, new_settings)

    def reload_model(self, model_type: str, settings: Dict[str, Any]) -> None:
        """モデルをリロードします"""
        self.model_loaded.clear()

        try:
            model_path = os.path.join(MODEL_DIR, settings['model_path'])
            self.llm_adapter = LlamaAdapter(model_path, settings)
            self.current_model = model_type
            self.model_loaded.set()
            print(f"{model_type}モデルをロードしました。モデルパス: {model_path}、GPUレイヤー数: {settings['n_gpu_layers']}")
        except Exception as e:
            print(f"{model_type}モデルのロード中にエラーが発生しました: {e}")
            import traceback
            traceback.print_exc()
            self.model_loaded.set()

    def get_current_settings(self, model_type: str) -> Dict[str, Any]:
        """現在の設定を取得します"""
        return {
            'model_path': self.settings[f'DEFAULT_{model_type.upper()}_MODEL'],
            'n_gpu_layers': self.settings[f'{model_type.lower()}_n_gpu_layers'],
            'temperature': self.settings[f'{model_type.lower()}_temperature'],
            'top_p': self.settings[f'{model_type.lower()}_top_p'],
            'top_k': self.settings[f'{model_type.lower()}_top_k'],
            'repeat_penalty': self.settings[f'{model_type.lower()}_rep_pen'],
            'n_ctx': self.settings[f'{model_type.lower()}_n_ctx']
        }

    def are_models_identical(self) -> bool:
        """CHATモデルとGENモデルの設定が同じかどうかを確認します"""
        return self.chat_model_settings == self.gen_model_settings

    def generate_response(self, input_str: str) -> str:
        """応答を生成します"""
        self.load_model('CHAT')
        if not self.model_loaded.wait(timeout=30) or not self.llm_adapter:
            return "モデルのロードに失敗しました。設定を確認してください。"
        
        try:
            if self.use_chat_format:
                chat_messages = self._prepare_chat_messages(input_str)
                response = self.llm_adapter.create_chat_completion(chat_messages, max_tokens=1000)
                res_text = response["choices"][0]["message"]["content"].strip()
            else:
                prompt = self._generate_prompt(input_str)
                res_text = self.llm_adapter.generate_text(prompt, max_tokens=1000)
            
            if not res_text:
                return "申し訳ありません。有効な応答を生成できませんでした。もう一度お試しください。"
            
            self._update_history(input_str, res_text)
            return res_text
        except Exception as e:
            print(f"レスポンス生成中にエラーが発生しました: {str(e)}")
            return f"レスポンス生成中にエラーが発生しました: {str(e)}"

    def generate_text(self, text: str, gen_characters: int, gen_token_multiplier: float, instruction: str) -> str:
        """テキストを生成します"""
        self.load_model('GEN')
        if not self.model_loaded.wait(timeout=30) or not self.llm_adapter:
            return "モデルのロードに失敗しました。設定を確認してください。"
        
        author_description = self.settings.get('gen_author_description', '')
        max_tokens = int(gen_characters * gen_token_multiplier)
        
        try:
            if self.use_chat_format:
                messages = [
                    {"role": "system", "content": author_description},
                    {"role": "user", "content": f"以下の指示に従ってテキストを生成してください：\n\n{instruction}\n\n生成するテキスト（目安は{gen_characters}文字）：\n\n{text}"}
                ]
                
                response = self.llm_adapter.create_chat_completion(messages, max_tokens)
                generated_text = response["choices"][0]["message"]["content"].strip()
            else:
                prompt = f"{author_description}\n\n以下の指示に従ってテキストを生成してください：\n\n{instruction}\n\n生成するテキスト（目安は{gen_characters}文字）：\n\n{text}\n\n生成されたテキスト："
                generated_text = self.llm_adapter.generate_text(prompt, max_tokens)
            
            if not generated_text:
                return "申し訳ありません。有効なテキストを生成できませんでした。もう一度お試しください。"
            
            return generated_text
        except Exception as e:
            print(f"テキスト生成中にエラーが発生しました: {str(e)}")
            return f"テキスト生成中にエラーが発生しました: {str(e)}"

    def set_chat_format(self, use_chat_format: bool) -> None:
        """チャットフォーマットを設定します"""
        self.use_chat_format = use_chat_format

    def _prepare_chat_messages(self, input_str: str) -> List[Dict[str, str]]:
        """チャットメッセージを準備します"""
        messages = [{"role": "system", "content": self.settings.get('chat_author_description', '')}]
        messages.extend(self.chat_history)
        messages.append({"role": "user", "content": input_str})
        return messages

    def _generate_prompt(self, input_str: str) -> str:
        """プロンプトを生成します"""
        prompt_template = """{{chat_author_description}}

{{chat_instructions}}

・キャラクターの回答例
{% for qa in example_qa %}
{{qa}}
{% endfor %}

・会話履歴
{% for history in histories %}
user: {{history.user}}
assistant: {{history.assistant}}
{% endfor %}

user: {{input_str}}
assistant:"""
        
        template = Template(prompt_template)
        return template.render(
            chat_author_description=self.settings.get('chat_author_description', ''),
            chat_instructions=self.settings.get('chat_instructions', ''),
            example_qa=self.settings.get('example_qa', []),
            histories=self.history,
            input_str=input_str
        )

    def _update_history(self, input_str: str, res_text: str) -> None:
            """履歴を更新します"""
            if self.use_chat_format:
                self.chat_history.append({"role": "user", "content": input_str})
                self.chat_history.append({"role": "assistant", "content": res_text})
            else:
                self.history.append({"user": input_str, "assistant": res_text})

    def load_character(self, filename: str) -> None:
        """キャラクター設定をロードします"""
        if isinstance(filename, list):
            filename = filename[0] if filename else ""
        self.settings = Settings.load_from_ini(filename)

    def reset(self) -> None:
        """履歴をリセットします"""
        self.history = []
        self.chat_history = []
        self.use_chat_format = False

class Settings:
    @staticmethod
    def _parse_config(config: configparser.ConfigParser) -> Dict[str, Any]:
        """設定ファイルを解析します"""
        settings = {}
        if 'Character' in config:
            settings['chat_author_description'] = config['Character'].get('chat_author_description', '')
            settings['chat_instructions'] = config['Character'].get('chat_instructions', '')
            settings['example_qa'] = config['Character'].get('example_qa', '').split('\n')
            settings['gen_author_description'] = config['Character'].get('gen_author_description', '')
        if 'Models' in config:
            settings['DEFAULT_CHAT_MODEL'] = config['Models'].get('DEFAULT_CHAT_MODEL', '')
            settings['DEFAULT_GEN_MODEL'] = config['Models'].get('DEFAULT_GEN_MODEL', '')
        if 'ChatParameters' in config:
            settings['chat_n_gpu_layers'] = int(config['ChatParameters'].get('n_gpu_layers', '-1'))
            settings['chat_temperature'] = float(config['ChatParameters'].get('temperature', '0.35'))
            settings['chat_top_p'] = float(config['ChatParameters'].get('top_p', '0.9'))
            settings['chat_top_k'] = int(config['ChatParameters'].get('top_k', '40'))
            settings['chat_rep_pen'] = float(config['ChatParameters'].get('repetition_penalty', '1.2'))
            settings['chat_n_ctx'] = int(config['ChatParameters'].get('n_ctx', '10000'))
        if 'GenerateParameters' in config:
            settings['gen_n_gpu_layers'] = int(config['GenerateParameters'].get('n_gpu_layers', '-1'))
            settings['gen_temperature'] = float(config['GenerateParameters'].get('temperature', '0.35'))
            settings['gen_top_p'] = float(config['GenerateParameters'].get('top_p', '0.9'))
            settings['gen_top_k'] = int(config['GenerateParameters'].get('top_k', '40'))
            settings['gen_rep_pen'] = float(config['GenerateParameters'].get('repetition_penalty', '1.2'))
            settings['gen_n_ctx'] = int(config['GenerateParameters'].get('n_ctx', '10000'))
        return settings

    @staticmethod
    def save_to_ini(settings: Dict[str, Any], filename: str) -> None:
        """設定をINIファイルに保存します"""
        config = configparser.ConfigParser()
        config['Character'] = {
            'chat_author_description': settings.get('chat_author_description', ''),
            'chat_instructions': settings.get('chat_instructions', ''),
            'example_qa': '\n'.join(settings.get('example_qa', [])),
            'gen_author_description': settings.get('gen_author_description', '')
        }
        config['Models'] = {
            'DEFAULT_CHAT_MODEL': settings.get('DEFAULT_CHAT_MODEL', ''),
            'DEFAULT_GEN_MODEL': settings.get('DEFAULT_GEN_MODEL', '')
        }
        config['ChatParameters'] = {
            'n_gpu_layers': str(settings.get('chat_n_gpu_layers', -1)),
            'temperature': str(settings.get('chat_temperature', 0.35)),
            'top_p': str(settings.get('chat_top_p', 0.9)),
            'top_k': str(settings.get('chat_top_k', 40)),
            'repetition_penalty': str(settings.get('chat_rep_pen', 1.2)),
            'n_ctx': str(settings.get('chat_n_ctx', 10000))
        }
        config['GenerateParameters'] = {
            'n_gpu_layers': str(settings.get('gen_n_gpu_layers', -1)),
            'temperature': str(settings.get('gen_temperature', 0.35)),
            'top_p': str(settings.get('gen_top_p', 0.9)),
            'top_k': str(settings.get('gen_top_k', 40)),
            'repetition_penalty': str(settings.get('gen_rep_pen', 1.2)),
            'n_ctx': str(settings.get('gen_n_ctx', 10000))
        }
        with open(filename, 'w', encoding='utf-8') as configfile:
            config.write(configfile)

    @staticmethod
    def create_default_ini(filename: str) -> None:
        """デフォルトのINIファイルを作成します"""
        default_settings = {
            'chat_author_description': "あなたは優秀な小説執筆アシスタントです。三幕構造や起承転結、劇中劇などのあらゆる小説理論や小説技法にも通じています。",
            'chat_instructions': "丁寧な敬語でアイディアのヒアリングしてください。物語をより面白くする提案、キャラクター造形の考察、世界観を膨らませる手伝いなどをお願いします。求められた時以外は基本、聞き役に徹してユーザー自身に言語化させるよう促してください。ユーザーのことは『ユーザー』と呼んでください。",
            'example_qa': [
            "user: キャラクターの設定について悩んでいます。",
            "assistant: キャラクター設定は物語の核となる重要な要素ですね。ユーザーが現在考えているキャラクターについて、簡単にご説明いただけますでしょうか？",
            "user: どんな設定を説明をしたらいいでしょうか？",
            "assistant: 例えば、年齢、性別、職業、性格の特徴などから始めていただけると、より具体的なアドバイスができるかと思います。",
            "user: プロットを書き出したいので、ヒアリングお願いします。",
            "assistant: 承知しました。ではまず『起承転結』の起から考えていきましょう。",
            "user: 読者を惹きこむ為のコツを提案してください",
            "assistant: 諸説ありますが、『謎・ピンチ・意外性』を冒頭に持ってくることが重要だと言います。",
            "user: プロットが面白いか自信がないので、考察のお手伝いをお願いします。",
            "assistant: プロットについてコメントをする前に、まずこの物語の『売り』について簡単に説明してください",
            ],
            'gen_author_description': 'あなたは新進気鋭の和風伝奇ミステリー小説家で、細やかな筆致と巧みな構成で若い世代にとても人気があります。',
            'DEFAULT_CHAT_MODEL': 'EZO-Common-9B-gemma-2-it.Q8_0.gguf',
            'DEFAULT_GEN_MODEL': 'EZO-Common-9B-gemma-2-it.Q8_0.gguf',
            'chat_n_gpu_layers': 120,
            'chat_temperature': 0.35,
            'chat_top_p': 0.9,
            'chat_top_k': 40,
            'chat_rep_pen': 1.2,
            'chat_n_ctx': 10000,
            'gen_n_gpu_layers': 120,
            'gen_temperature': 0.35,
            'gen_top_p': 0.9,
            'gen_top_k': 40,
            'gen_rep_pen': 1.2,
            'gen_n_ctx': 10000
        }
        Settings.save_to_ini(default_settings, filename)

    @staticmethod
    def load_from_ini(filename: str) -> Dict[str, Any]:
        """INIファイルから設定をロードします"""
        config = configparser.ConfigParser()
        config.read(filename, encoding='utf-8')
        return Settings._parse_config(config)

# グローバル変数
character_maker: CharacterMaker = None
model_files: List[str] = []
temp_settings: Dict[str, Dict[str, Any]] = {}

# チャット関連関数
def chat_with_character(message: str, history: List[List[str]]) -> str:
    """キャラクターとチャットします"""
    if character_maker.use_chat_format:
        character_maker.chat_history = [{"role": "user" if i % 2 == 0 else "assistant", "content": msg} for i, msg in enumerate(sum(history, []))]
    else:
        character_maker.history = [{"user": h[0], "assistant": h[1]} for h in history]
    return character_maker.generate_response(message)

def chat_with_character_stream(message: str, history: List[List[str]]) -> str:
    """キャラクターとのチャットをストリーム形式で行います"""
    if character_maker.use_chat_format:
        character_maker.chat_history = [{"role": "user" if i % 2 == 0 else "assistant", "content": msg} for i, msg in enumerate(sum(history, []))]
    else:
        character_maker.history = [{"user": h[0], "assistant": h[1]} for h in history]
    response = character_maker.generate_response(message)
    for i in range(len(response)):
        time.sleep(0.05)  # 各文字の表示間隔を調整
        yield response[:i+1]

def clear_chat() -> List:
    """チャットをクリアします"""
    character_maker.reset()
    return []

# ログ関連関数
def list_log_files() -> List[str]:
    """ログファイルのリストを取得します"""
    logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    if not os.path.exists(logs_dir):
        return []
    return [f for f in os.listdir(logs_dir) if f.endswith('.csv')]

def load_chat_log(file_name: str) -> List[List[str]]:
    """チャットログを読み込みます"""
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", file_name)
    chat_history = []
    with open(file_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # ヘッダーをスキップ
        for row in reader:
            if len(row) == 2:
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
    logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    file_path = os.path.join(logs_dir, filename)
    
    with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Role", "Message"])
        for user_message, assistant_message in chat_history:
            if user_message:
                writer.writerow(["user", user_message])
            if assistant_message:
                writer.writerow(["assistant", assistant_message])
    
    return f"チャットログが {file_path} に保存されました。"

def resume_chat_from_log(chat_history: List[List[str]]) -> gr.update:
    """ログからチャットを再開します"""
    # チャットボットのUIを更新
    chatbot_ui = gr.update(value=chat_history)
    
    # LLMの履歴を更新
    character_maker.history = [{"user": h[0], "assistant": h[1]} for h in chat_history if h[0] is not None and h[1] is not None]
    
    return chatbot_ui

def update_temp_setting(section: str, key: str, value: Any) -> str:
    """一時的な設定を更新します"""
    global temp_settings
    if section not in temp_settings:
        temp_settings[section] = {}
    temp_settings[section][key] = value
    return f"{section}セクションの{key}を更新しました。適用ボタンを押すと設定が保存されます。"

def build_model_settings(config: configparser.ConfigParser, section: str, output: gr.components.Textbox) -> List[gr.components.Component]:
    """モデル設定のUIを構築します"""
    model_settings = []

    for key in ['DEFAULT_CHAT_MODEL', 'DEFAULT_GEN_MODEL']:
        if key in config[section]:
            with gr.Row():
                dropdown = gr.Dropdown(
                    label=key,
                    choices=ModelManager.get_model_files(),
                    value=config[section][key]
                )
                refresh_button = gr.Button("更新", size="sm")
                status_message = gr.Markdown()
            
            def update_dropdown(current_value):
                model_files = ModelManager.get_model_files()
                if current_value not in model_files:
                    model_files.insert(0, current_value)
                    status = f"現在の{key}（{current_value}）が見つかりません。ダウンロードしてください。"
                else:
                    status = "モデルリストを更新しました。"
                return gr.update(choices=model_files, value=current_value), status

            refresh_button.click(
                fn=update_dropdown,
                inputs=[dropdown],
                outputs=[dropdown, status_message]
            )
            
            dropdown.change(
                partial(update_temp_setting, 'Models', key),
                inputs=[dropdown],
                outputs=[output]
            )
            
            model_settings.extend([dropdown, refresh_button, status_message])

    return model_settings

def apply_settings() -> str:
    """設定を適用します"""
    global temp_settings, character_maker
    settings_changed = False
    
    config = Config(DEFAULT_INI_FILE)
    
    for section, settings in temp_settings.items():
        for key, value in settings.items():
            old_value = config.get(section, key)
            if str(value) != str(old_value):
                config.update(section, key, str(value))
                settings_changed = True
    
    if not settings_changed:
        return "設定に変更はありませんでした。"
    
    # 設定を更新
    character_maker.settings = Settings._parse_config(config.config)
    
    # モデルのリロードをトリガー（実際のリロードは次の操作時に行われる）
    character_maker.chat_model_settings = None
    character_maker.gen_model_settings = None
    
    # temp_settings をクリア
    temp_settings.clear()
    
    return "設定をiniファイルに保存し、アプリケーションに反映しました。次回の操作時に新しい設定が適用されます。"

# Gradioインターフェース
def build_gradio_interface() -> gr.Blocks:
    """Gradioインターフェースを構築します"""
    with gr.Blocks() as iface:
        gr.HTML("""
        <div style="background-color: #f0f0f0; padding: 10px; margin-bottom: 10px; border-radius: 5px;">
            <strong>注意：</strong>念のため、NSFW創作用途の場合はモデルを設定タブから、「EZO-Common-9B-gemma-2-it.Q8_0.gguf」→「Mistral-Nemo-Instruct-2407-Q8_0.gguf」に変更推奨です。
            <br>
            <a href="https://note.com/eurekachan/n/nd05d6307fead" target="_blank" style="color: #007bff; text-decoration: underline;">
                参考情報はこちら
            </a>
        </div>
        """)

        gr.HTML("""
        <style>
        #chatbot, #chatbot_read {
            resize: both;
            overflow: auto;
            min-height: 100px;
            max-height: 80vh;
        }
        </style>
        """)
        tabs = gr.Tabs()
        with tabs:
            with gr.Tab("チャット", id="chat_tab") as chat_tab:
                chatbot = gr.Chatbot(elem_id="chatbot")
                chat_interface = gr.ChatInterface(
                    chat_with_character_stream,
                    chatbot=chatbot,
                    textbox=gr.Textbox(placeholder="メッセージを入力してください...", container=False, scale=7),
                    theme="soft",
                    submit_btn="送信",
                    stop_btn="停止",
                    retry_btn="もう一度生成",
                    undo_btn="前のメッセージを取り消す",
                    clear_btn="チャットをクリア",
                )
                
                with gr.Row():
                    save_log_button = gr.Button("チャットログを保存")
                
                save_log_output = gr.Textbox(label="保存状態")
                
                save_log_button.click(
                    save_chat_log,
                    inputs=[chatbot],
                    outputs=[save_log_output]
                )

            with gr.Tab("文章生成"):
                with gr.Row():
                    with gr.Column(scale=2):
                        instruction_type = gr.Dropdown(
                            choices=["自由入力", "推敲", "プロット作成", "あらすじ作成", "地の文追加"],
                            label="指示タイプ",
                            value="自由入力"
                        )
                        gen_instruction = gr.Textbox(
                            label="指示",
                            value="",
                            lines=3
                        )
                        gen_input_text = gr.Textbox(lines=5, label="処理されるテキストを入力してください")  
                        gen_input_char_count = gr.HTML(value="文字数: 0")                                        
                    with gr.Column(scale=1):
                        gen_characters = gr.Slider(minimum=10, maximum=10000, value=500, step=10, label="出力文字数", info="出力文字数の目安")
                        gen_token_multiplier = gr.Slider(minimum=0.35, maximum=3, value=1.75, step=0.01, label="文字/トークン数倍率", info="文字/最大トークン数倍率")
                
                generate_button = gr.Button("文章生成開始")
                generated_output = gr.Textbox(label="生成された文章")
            
                generate_button.click(
                    character_maker.generate_text,
                    inputs=[gen_input_text, gen_characters, gen_token_multiplier, gen_instruction],
                    outputs=[generated_output]
                )

                def update_instruction(choice: str) -> str:
                    """指示タイプに応じて指示テキストを更新します"""
                    instructions = {
                        "自由入力": "",
                        "推敲": "以下のテキストを推敲してください。原文の文体や特徴的な表現は保持しつつ、必要に応じて微調整を加えてください。文章の流れを自然にし、表現を洗練させることが目標ですが、元の雰囲気や個性を損なわないよう注意してください。",
                        "プロット作成": "以下のテキストをプロットにしてください。起承転結に分割すること。",
                        "あらすじ作成": "以下のテキストをあらすじにして、簡潔にまとめてください。",
                        "地の文追加": "以下のテキストの地の文を増やして、描写を膨らませるように推敲してください。文章の流れを自然にし、表現を洗練させることが目標ですが、なるべく元の文の意味や流れは残してください、"
                    }
                    return instructions.get(choice, "")

                instruction_type.change(
                    update_instruction,
                    inputs=[instruction_type],
                    outputs=[gen_instruction]
                )

                def update_char_count(text: str) -> str:
                    """文字数を更新します"""
                    return f"文字数: {len(text)}"

                gen_input_text.change(
                    update_char_count,
                    inputs=[gen_input_text],
                    outputs=[gen_input_char_count]
                )

            with gr.Tab("ログ閲覧", id="log_view_tab") as log_view_tab:
                gr.Markdown("## チャットログ閲覧")
                chatbot_read = gr.Chatbot(elem_id="chatbot_read")
                log_file_dropdown = gr.Dropdown(label="ログファイル選択", choices=list_log_files())
                refresh_log_list_button = gr.Button("ログファイルリストを更新")
                resume_chat_button = gr.Button("選択したログから会話を再開")

                def update_log_dropdown() -> gr.update:
                    """ログファイルのドロップダウンを更新します"""
                    return gr.update(choices=list_log_files())

                def load_and_display_chat_log(file_name: str) -> gr.update:
                    """チャットログを読み込んで表示します"""
                    chat_history = load_chat_log(file_name)
                    return gr.update(value=chat_history)

                refresh_log_list_button.click(
                    update_log_dropdown,
                    outputs=[log_file_dropdown]
                )

                log_file_dropdown.change(
                    load_and_display_chat_log,
                    inputs=[log_file_dropdown],
                    outputs=[chatbot_read]
                )

                def resume_chat_and_switch_tab(chat_history: List[List[str]]) -> Tuple[gr.update, gr.update]:
                    """ログからチャットを再開し、タブを切り替えます"""
                    chatbot_ui = resume_chat_from_log(chat_history)
                    return chatbot_ui, gr.update(selected="chat_tab")

                resume_chat_button.click(
                    resume_chat_and_switch_tab,
                    inputs=[chatbot_read],
                    outputs=[chatbot, tabs]
                )

            with gr.Tab("設定"):
                output = gr.Textbox(label="更新状態")
                
                config = Config(DEFAULT_INI_FILE)

                with gr.Column():
                    gr.Markdown("### モデル設定")
                    model_settings = build_model_settings(config.config, "Models", output)

                    gr.Markdown("### チャット設定")
                    for key in ['chat_author_description', 'chat_instructions', 'example_qa']:
                        if key == 'example_qa':
                            input_component = gr.TextArea(label=key, value=config.get('Character', key, ''), lines=10)
                        else:
                            input_component = gr.TextArea(label=key, value=config.get('Character', key, ''), lines=5)
                        input_component.change(
                            partial(update_temp_setting, 'Character', key),
                            inputs=[input_component],
                            outputs=[output]
                        )

                    gr.Markdown("### 文章生成設定")
                    key = 'gen_author_description'
                    input_component = gr.TextArea(label=key, value=config.get('Character', key, ''), lines=5)
                    input_component.change(
                        partial(update_temp_setting, 'Character', key),
                        inputs=[input_component],
                        outputs=[output]
                    )

                    gr.Markdown("### チャットパラメータ設定")
                    for key in ['n_gpu_layers', 'temperature', 'top_p', 'top_k', 'repetition_penalty', 'n_ctx']:
                        value = config.get('ChatParameters', key, '0')
                        if key == 'n_gpu_layers':
                            input_component = gr.Slider(label=key, value=int(value), minimum=-1, maximum=255, step=1)
                        elif key in ['temperature', 'top_p', 'repetition_penalty']:
                            input_component = gr.Slider(label=key, value=float(value), minimum=0.0, maximum=1.0, step=0.05)
                        elif key == 'top_k':
                            input_component = gr.Slider(label=key, value=int(value), minimum=1, maximum=200, step=1)
                        elif key == 'n_ctx':
                            input_component = gr.Slider(label=key, value=int(value), minimum=10000, maximum=100000, step=1000)
                        else:
                            input_component = gr.Textbox(label=key, value=value)
                        
                        input_component.change(
                            partial(update_temp_setting, 'ChatParameters', key),
                            inputs=[input_component],
                            outputs=[output]
                        )

                    gr.Markdown("### 文章生成パラメータ設定")
                    for key in ['n_gpu_layers', 'temperature', 'top_p', 'top_k', 'repetition_penalty', 'n_ctx']:
                        value = config.get('GenerateParameters', key, '0')
                        if key == 'n_gpu_layers':
                            input_component = gr.Slider(label=key, value=int(value), minimum=-1, maximum=255, step=1)
                        elif key in ['temperature', 'top_p', 'repetition_penalty']:
                            input_component = gr.Slider(label=key, value=float(value), minimum=0.0, maximum=1.0, step=0.05)
                        elif key == 'top_k':
                            input_component = gr.Slider(label=key, value=int(value), minimum=1, maximum=200, step=1)
                        elif key == 'n_ctx':
                            input_component = gr.Slider(label=key, value=int(value), minimum=10000, maximum=100000, step=1000)
                        else:
                            input_component = gr.Textbox(label=key, value=value)
                        
                        input_component.change(
                            partial(update_temp_setting, 'GenerateParameters', key),
                            inputs=[input_component],
                            outputs=[output]
                        )

                    apply_ini_settings_button = gr.Button("設定を適用")
                    apply_ini_settings_button.click(
                        apply_settings,
                        outputs=[output]
                    )
    return iface

async def start_gradio() -> None:
    """Gradioアプリケーションを起動します"""
    global character_maker, model_files

    if not os.path.exists(DEFAULT_INI_FILE):
        print(f"{DEFAULT_INI_FILE} が見つかりません。デフォルト設定で作成します。")
        Settings.create_default_ini(DEFAULT_INI_FILE)

    config = Config(DEFAULT_INI_FILE)
    settings = Settings._parse_config(config.config)

    model_manager = ModelManager(MODEL_DIR)
    llm_adapter = LlamaAdapter(os.path.join(MODEL_DIR, settings['DEFAULT_CHAT_MODEL']), {
        'n_ctx': settings['chat_n_ctx'],
        'n_gpu_layers': settings['chat_n_gpu_layers'],
        'temperature': settings['chat_temperature'],
        'top_p': settings['chat_top_p'],
        'top_k': settings['chat_top_k'],
        'repeat_penalty': settings['chat_rep_pen'],
    })
    character_maker = CharacterMaker(llm_adapter, settings)
    model_files = model_manager.get_model_files()
    
    demo = build_gradio_interface()

    ip_address = NetworkUtils.get_ip_address()
    starting_port = 7860
    port = NetworkUtils.find_available_port(starting_port)
    print(f"サーバーのアドレス: http://{ip_address}:{port}")
    
    demo.queue()
    demo.launch(
        server_name='0.0.0.0', 
        server_port=port,
        share=False,
        favicon_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "custom.html")
    )

if __name__ == "__main__":
    asyncio.run(start_gradio())
