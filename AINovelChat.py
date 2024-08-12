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

# 定数
DEFAULT_INI_FILE = 'settings.ini'
MODEL_FILE_EXTENSION = '.gguf'

# パスの設定
if getattr(sys, 'frozen', False):
    BASE_PATH = os.path.dirname(sys.executable)
    MODEL_DIR = os.path.join(os.path.dirname(BASE_PATH), "AI-NovelAssistant", "models")
else:
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))
    MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

class ConfigManager:
    @staticmethod
    def load_settings(filename):
        config = configparser.ConfigParser()
        config.read(filename, encoding='utf-8')
        return config

    @staticmethod
    def save_settings(config, filename):
        with open(filename, 'w', encoding='utf-8') as configfile:
            config.write(configfile)

    @staticmethod
    def update_setting(section, key, value, filename):
        config = ConfigManager.load_settings(filename)
        config[section][key] = value
        ConfigManager.save_settings(config, filename)
        return f"設定を更新しました: [{section}] {key} = {value}"

class ModelManager:
    @staticmethod
    def get_model_files():
        return [f for f in os.listdir(MODEL_DIR) if f.endswith(MODEL_FILE_EXTENSION)]

    @staticmethod
    def update_model_dropdown(config, section, key):
        current_value = config[section][key]
        model_files = ModelManager.get_model_files()
        
        if current_value not in model_files:
            download_message = f"現在の{key}（{current_value}）が見つかりません。ダウンロードしてください。"
            model_files.insert(0, current_value)
        else:
            download_message = ""
        
        return model_files, current_value, download_message

class NetworkUtils:
    @staticmethod
    def get_ip_address():
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            try:
                s.connect(('10.255.255.255', 1))
                return s.getsockname()[0]
            except Exception:
                return '127.0.0.1'

    @staticmethod
    def find_available_port(starting_port):
        port = starting_port
        while NetworkUtils.is_port_in_use(port):
            print(f"Port {port} is in use, trying next one.")
            port += 1
        return port

    @staticmethod
    def is_port_in_use(port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) == 0

class Settings:
    @staticmethod
    def _parse_config(config):
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
            settings['chat_n_gpu_layers'] = int(config['ChatParameters'].get('n_gpu_layers', '0'))
            settings['chat_temperature'] = float(config['ChatParameters'].get('temperature', '0.5'))
            settings['chat_top_p'] = float(config['ChatParameters'].get('top_p', '0.7'))
            settings['chat_top_k'] = int(config['ChatParameters'].get('top_k', '80'))
            settings['chat_rep_pen'] = float(config['ChatParameters'].get('repetition_penalty', '1.2'))
        if 'GenerateParameters' in config:
            settings['gen_n_gpu_layers'] = int(config['GenerateParameters'].get('n_gpu_layers', '0'))
            settings['gen_temperature'] = float(config['GenerateParameters'].get('temperature', '0.35'))
            settings['gen_top_p'] = float(config['GenerateParameters'].get('top_p', '0.9'))
            settings['gen_top_k'] = int(config['GenerateParameters'].get('top_k', '40'))
            settings['gen_rep_pen'] = float(config['GenerateParameters'].get('repetition_penalty', '1.2'))
        return settings

    @staticmethod
    def save_to_ini(settings, filename):
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
            'n_gpu_layers': str(settings.get('chat_n_gpu_layers', 0)),
            'temperature': str(settings.get('chat_temperature', 0.5)),
            'top_p': str(settings.get('chat_top_p', 0.7)),
            'top_k': str(settings.get('chat_top_k', 80)),
            'repetition_penalty': str(settings.get('chat_rep_pen', 1.2))
        }
        config['GenerateParameters'] = {
            'n_gpu_layers': str(settings.get('gen_n_gpu_layers', 0)),
            'temperature': str(settings.get('gen_temperature', 0.35)),
            'top_p': str(settings.get('gen_top_p', 0.9)),
            'top_k': str(settings.get('gen_top_k', 40)),
            'repetition_penalty': str(settings.get('gen_rep_pen', 1.2))
        }
        ConfigManager.save_settings(config, filename)

    @staticmethod
    def create_default_ini(filename):
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
            'DEFAULT_CHAT_MODEL': 'Ninja-v1-RP-expressive-v2_Q4_K_M.gguf',
            'DEFAULT_GEN_MODEL': 'Mistral-Nemo-Instruct-2407-Q8_0.gguf',
            'chat_n_gpu_layers': 0,
            'chat_temperature': 0.5,
            'chat_top_p': 0.7,
            'chat_top_k': 80,
            'chat_rep_pen': 1.2,
            'gen_n_gpu_layers': 0,
            'gen_temperature': 0.35,
            'gen_top_p': 0.9,
            'gen_top_k': 40,
            'gen_rep_pen': 1.2
        }
        Settings.save_to_ini(default_settings, filename)

    @staticmethod
    def load_from_ini(filename):
        config = ConfigManager.load_settings(filename)
        return Settings._parse_config(config)

class GenTextParams:
    def __init__(self):
        self.gen_n_gpu_layers = 0
        self.gen_temperature = 0.35
        self.gen_top_p = 1.0
        self.gen_top_k = 40
        self.gen_rep_pen = 1.0
        self.chat_n_gpu_layers = 0
        self.chat_temperature = 0.5
        self.chat_top_p = 0.7
        self.chat_top_k = 80
        self.chat_rep_pen = 1.2

    def update_generate_parameters(self, n_gpu_layers, temperature, top_p, top_k, rep_pen):
        self.gen_n_gpu_layers = n_gpu_layers
        self.gen_temperature = temperature
        self.gen_top_p = top_p
        self.gen_top_k = top_k
        self.gen_rep_pen = rep_pen

    def update_chat_parameters(self, n_gpu_layers, temperature, top_p, top_k, rep_pen):
        self.chat_n_gpu_layers = n_gpu_layers
        self.chat_temperature = temperature
        self.chat_top_p = top_p
        self.chat_top_k = top_k
        self.chat_rep_pen = rep_pen

class LlamaAdapter:
    def __init__(self, model_path, params, n_gpu_layers):
        self.llm = Llama(model_path=model_path, n_ctx=10000, n_gpu_layers=n_gpu_layers)
        self.params = params

    def generate_text(self, text, author_description, gen_characters, gen_token_multiplier, instruction):
        max_tokens = gen_characters * gen_token_multiplier
        response = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": author_description},
                {"role": "user", "content": f"{instruction}\n出力文字数目安は{gen_characters}字です：\n\n{text}"},
            ],
            max_tokens=max_tokens, temperature=self.params.gen_temperature, top_p=self.params.gen_top_p,
            top_k=self.params.gen_top_k, repeat_penalty=self.params.gen_rep_pen,
        )
        return response["choices"][0]["message"]["content"].strip()

    def generate(self, prompt, max_new_tokens=10000):
        return self.llm(prompt, temperature=self.params.chat_temperature, max_tokens=max_new_tokens,
                        top_p=self.params.chat_top_p, top_k=self.params.chat_top_k,
                        repeat_penalty=self.params.chat_rep_pen, stop=["user:", "・会話履歴", "<END>"])

class CharacterMaker:
    def __init__(self):
        self.llama = None
        self.history = []
        self.settings = None
        self.model_loaded = threading.Event()
        self.current_model = None
        self.model_lock = threading.Lock()

    def load_model(self, model_type):
        with self.model_lock:
            if self.current_model == model_type:
                return

            self.model_loaded.clear()
            if self.llama:
                del self.llama
                self.llama = None

            try:
                model_path = os.path.join(MODEL_DIR, self.settings[f'DEFAULT_{model_type.upper()}_MODEL'])
                n_gpu_layers = self.settings[f'{model_type.lower()}_n_gpu_layers']
                self.llama = LlamaAdapter(model_path, params, n_gpu_layers)
                self.current_model = model_type
                self.model_loaded.set()
                print(f"{model_type} モデル {model_path} のロードが完了しました。(n_gpu_layers: {n_gpu_layers})")
            except Exception as e:
                print(f"{model_type} モデルのロード中にエラーが発生しました: {str(e)}")
                self.model_loaded.set()

    def generate_response(self, input_str):
        self.load_model('CHAT')
        if not self.model_loaded.wait(timeout=30) or not self.llama:
            return "モデルのロードに失敗しました。設定を確認してください。"
        
        prompt = self._generate_prompt(input_str)
        res = self.llama.generate(prompt, max_new_tokens=1000)
        res_text = res["choices"][0]["text"]
        self.history.append({"user": input_str, "assistant": res_text})
        return res_text

    def generate_text(self, text, gen_characters, gen_token_multiplier, instruction):
        self.load_model('GEN')
        if not self.model_loaded.wait(timeout=30) or not self.llama:
            return "モデルのロードに失敗しました。設定を確認してください。"
        
        author_description = self.settings.get('gen_author_description', '')
        return self.llama.generate_text(text, author_description, gen_characters, gen_token_multiplier, instruction)

    def make_prompt(self, input_str: str):
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
    
    def _generate_prompt(self, input_str: str):
        return self.make_prompt(input_str)

    def load_character(self, filename):
        if isinstance(filename, list):
            filename = filename[0] if filename else ""
        self.settings = Settings.load_from_ini(filename)

    def reset(self):
        self.history = []

# グローバル変数
params = GenTextParams()
character_maker = CharacterMaker()
model_files = ModelManager.get_model_files()

# チャット関連関数
def chat_with_character(message, history):
    character_maker.history = [{"user": h[0], "assistant": h[1]} for h in history]
    return character_maker.generate_response(message)

def chat_with_character_stream(message, history):
    character_maker.history = [{"user": h[0], "assistant": h[1]} for h in history]
    response = character_maker.generate_response(message)
    for i in range(len(response)):
        time.sleep(0.05)  # 各文字の表示間隔を調整
        yield response[:i+1]

def clear_chat():
    character_maker.reset()
    return []

# ログ関連関数
def list_log_files():
    logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    if not os.path.exists(logs_dir):
        return []
    return [f for f in os.listdir(logs_dir) if f.endswith('.csv')]

def load_chat_log(file_name):
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", file_name)
    chat_history = []
    with open(file_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
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

def save_chat_log(chat_history):
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

def resume_chat_from_log(chat_history):
    # チャットボットのUIを更新
    chatbot_ui = gr.update(value=chat_history)
    
    # LLMの履歴を更新
    character_maker.history = [{"user": h[0], "assistant": h[1]} for h in chat_history if h[0] is not None and h[1] is not None]
    
    return chatbot_ui

def build_model_settings(config, section, output):
    model_settings = []
    section_temp_settings = {}

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
            
            def update_temp_setting(value):
                section_temp_settings[key] = value
                return f"{key}の選択を{value}に更新しました。適用ボタンを押すと設定が保存されます。"

            dropdown.change(
                update_temp_setting,
                inputs=[dropdown],
                outputs=[output]
            )
            
            model_settings.extend([dropdown, refresh_button, status_message])
    
    return model_settings, section_temp_settings

# Gradioインターフェース
def build_gradio_interface():
    def apply_settings():
        for section, settings in temp_settings.items():
            for key, value in settings.items():
                ConfigManager.update_setting(section, key, str(value), DEFAULT_INI_FILE)
        
        # iniファイルを再読み込み
        new_config = ConfigManager.load_settings(DEFAULT_INI_FILE)
        
        # 設定を更新
        character_maker.settings = Settings._parse_config(new_config)
        
        # パラメータを更新
        if 'ChatParameters' in new_config:
            params.update_chat_parameters(
                int(new_config['ChatParameters'].get('n_gpu_layers', '0')),
                float(new_config['ChatParameters'].get('temperature', '0.5')),
                float(new_config['ChatParameters'].get('top_p', '0.7')),
                int(new_config['ChatParameters'].get('top_k', '80')),
                float(new_config['ChatParameters'].get('repetition_penalty', '1.2'))
            )
        if 'GenerateParameters' in new_config:
            params.update_generate_parameters(
                int(new_config['GenerateParameters'].get('n_gpu_layers', '0')),
                float(new_config['GenerateParameters'].get('temperature', '0.35')),
                float(new_config['GenerateParameters'].get('top_p', '0.9')),
                int(new_config['GenerateParameters'].get('top_k', '40')),
                float(new_config['GenerateParameters'].get('repetition_penalty', '1.2'))
            )
        
        return "設定をiniファイルに保存し、アプリケーションに反映しました。"
    
    with gr.Blocks() as demo:
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
                            choices=["自由入力", "推敲", "プロット作成", "あらすじ作成"],
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
                        gen_token_multiplier = gr.Slider(minimum=0.5, maximum=3, value=1.75, step=0.01, label="文字/トークン数倍率", info="文字/最大トークン数倍率")
                
                generate_button = gr.Button("文章生成開始")
                generated_output = gr.Textbox(label="生成された文章")
            
                generate_button.click(
                    character_maker.generate_text,
                    inputs=[gen_input_text, gen_characters, gen_token_multiplier, gen_instruction],
                    outputs=[generated_output]
                )

                def update_instruction(choice):
                    instructions = {
                        "自由入力": "",
                        "推敲": "以下のテキストを推敲してください。原文の文体や特徴的な表現は保持しつつ、必要に応じて微調整を加えてください。文章の流れを自然にし、表現を洗練させることが目標ですが、元の雰囲気や個性を損なわないよう注意してください。",
                        "プロット作成": "以下のテキストをプロットにしてください。起承転結に分割すること。",
                        "あらすじ作成": "以下のテキストをあらすじにして、簡潔にまとめて下さい。",
                    }
                    return instructions.get(choice, "")

                instruction_type.change(
                    update_instruction,
                    inputs=[instruction_type],
                    outputs=[gen_instruction]
                )

                def update_char_count(text):
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

                def update_log_dropdown():
                    return gr.update(choices=list_log_files())

                def load_and_display_chat_log(file_name):
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

                def resume_chat_and_switch_tab(chat_history):
                    chatbot_ui = resume_chat_from_log(chat_history)
                    return chatbot_ui, gr.update(selected="chat_tab")

                resume_chat_button.click(
                    resume_chat_and_switch_tab,
                    inputs=[chatbot_read],
                    outputs=[chatbot, tabs]
                )

            with gr.Tab("設定"):
                output = gr.Textbox(label="更新状態")
                
                temp_settings = {}  # 一時的な設定を保存するための辞書
                
                def update_temp_setting(section, key, value):
                    if section not in temp_settings:
                        temp_settings[section] = {}
                    temp_settings[section][key] = value
                    return f"{section}セクションの{key}を更新しました。適用ボタンを押すと設定が保存されます。"
                
                config = ConfigManager.load_settings(DEFAULT_INI_FILE)

                with gr.Column():
                    gr.Markdown("### モデル設定")
                    model_settings, section_temp_settings = build_model_settings(config, "Models", output)
                    temp_settings.update(section_temp_settings)

                    gr.Markdown("### チャット設定")
                    for key in ['chat_author_description', 'chat_instructions', 'example_qa']:
                        if key == 'example_qa':
                            input_component = gr.TextArea(label=key, value=config['Character'].get(key, ''), lines=10)
                        else:
                            input_component = gr.TextArea(label=key, value=config['Character'].get(key, ''), lines=5)
                        input_component.change(
                            partial(update_temp_setting, 'Character', key),
                            inputs=[input_component],
                            outputs=[output]
                        )

                    gr.Markdown("### 文章生成設定")
                    key = 'gen_author_description'
                    input_component = gr.TextArea(label=key, value=config['Character'].get(key, ''), lines=5)
                    input_component.change(
                        partial(update_temp_setting, 'Character', key),
                        inputs=[input_component],
                        outputs=[output]
                    )

                    gr.Markdown("### チャットパラメータ設定")
                    for key, value in config['ChatParameters'].items():
                        if key == 'n_gpu_layers':
                            input_component = gr.Slider(label=key, value=int(value), minimum=-1, maximum=255, step=1)
                        elif key in ['temperature', 'top_p', 'repetition_penalty']:
                            input_component = gr.Slider(label=key, value=float(value), minimum=0.0, maximum=1.0, step=0.05)
                        elif key == 'top_k':
                            input_component = gr.Slider(label=key, value=int(value), minimum=1, maximum=200, step=1)
                        else:
                            input_component = gr.Textbox(label=key, value=value)
                        
                        input_component.change(
                            partial(update_temp_setting, 'ChatParameters', key),
                            inputs=[input_component],
                            outputs=[output]
                        )

                    gr.Markdown("### 文章生成パラメータ設定")
                    for key, value in config['GenerateParameters'].items():
                        if key == 'n_gpu_layers':
                            input_component = gr.Slider(label=key, value=int(value), minimum=-1, maximum=255, step=1)
                        elif key in ['temperature', 'top_p', 'repetition_penalty']:
                            input_component = gr.Slider(label=key, value=float(value), minimum=0.0, maximum=1.0, step=0.05)
                        elif key == 'top_k':
                            input_component = gr.Slider(label=key, value=int(value), minimum=1, maximum=200, step=1)
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

            apply_ini_settings_button = gr.Button("設定を適用")
            apply_ini_settings_button.click(
                apply_settings,
                outputs=[output]
            )

    return demo

async def start_gradio():
    if not os.path.exists(DEFAULT_INI_FILE):
        print(f"{DEFAULT_INI_FILE} が見つかりません。デフォルト設定で作成します。")
        Settings.create_default_ini(DEFAULT_INI_FILE)

    config = ConfigManager.load_settings(DEFAULT_INI_FILE)
    settings = Settings._parse_config(config)

    character_maker.settings = settings
    character_maker.load_character(DEFAULT_INI_FILE)
    
    # パラメータの初期化
    params.update_chat_parameters(
        settings['chat_n_gpu_layers'],
        settings['chat_temperature'],
        settings['chat_top_p'],
        settings['chat_top_k'],
        settings['chat_rep_pen']
    )
    params.update_generate_parameters(
        settings['gen_n_gpu_layers'],
        settings['gen_temperature'],
        settings['gen_top_p'],
        settings['gen_top_k'],
        settings['gen_rep_pen']
    )
    
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
