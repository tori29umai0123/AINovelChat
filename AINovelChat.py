import os
import sys
import time
from typing import List, Dict, Any, Tuple
import gradio as gr
from functools import partial
import asyncio

from utils.config import Config
from utils.character_maker import CharacterMaker
from utils.model_manager import ModelManager
from utils.network_utils import NetworkUtils
from utils.settings import Settings
from utils.logger import Logger
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

# グローバル変数
temp_settings: Dict[str, Dict[str, Any]] = {}

# チャット関連関数
def chat_with_character_stream(message: str, history: List[List[str]],character_maker: CharacterMaker) -> str:
    """キャラクターとのチャットをストリーム形式で行います"""
    if character_maker.use_cohere:
        character_maker.chat_history = [{"role": "USER" if i % 2 == 0 else "CHATBOT", "message": msg} for i, msg in enumerate(sum(history, []))]
    elif character_maker.use_chat_format:
        character_maker.chat_history = [{"role": "user" if i % 2 == 0 else "assistant", "content": msg} for i, msg in enumerate(sum(history, []))]
    else:
        character_maker.history = [{"user": h[0], "assistant": h[1]} for h in history]
    
    print(f"チャット生成開始 - 現在のモデル: {character_maker.current_model_info}")
    response = character_maker.generate_response(message)
    for i in range(len(response)):
        time.sleep(0.05)  # 各文字の表示間隔を調整
        yield response[:i+1]
    print(f"チャット生成完了 - 現在のモデル: {character_maker.current_model_info}")
# ログ関連関数


def resume_chat_from_log(chat_history: List[List[str]],character_maker: CharacterMaker) -> gr.update:
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

def apply_settings(character_maker: CharacterMaker) -> str:
    global temp_settings
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
    new_settings = Settings._parse_config(config.config)
    character_maker.update_settings(new_settings)
    
    # temp_settings をクリア
    temp_settings.clear()
    
    return "設定をiniファイルに保存し、アプリケーションに反映しました。次回の生成時に新しい設定が適用されます。"
class ChatApplicationGUI:
# Gradioインターフェース
    def build_gradio_interface(model_manager: ModelManager,character_maker: CharacterMaker) -> gr.Blocks:
        """Gradioインターフェースを構築します"""
        with gr.Blocks() as iface:
            gr.HTML("""
            <div style="background-color: #f0f0f0; padding: 10px; margin-bottom: 10px; border-radius: 5px;">
                <strong>注意：</strong>使用モデルの規約に従って生成してください。
                <br>NSFW創作はMistral-Nemo-Instruct-2407-Q8_0.ggufがおすすめです
                <br>各種規約情報は以下のリンク：
                <br>・<a href="https://note.com/eurekachan/n/nd05d6307fead" target="_blank" style="color: #007bff;">EZO-Common-9B-gemma-2-it.Q8_0</a>
                <br>・<a href="https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md" target="_blank" style="color: #007bff;">Mistral-Nemo-Instruct-2407-Q8_0</a>
                <br>・<a href="https://docs.cohere.com/docs/c4ai-acceptable-use-policy" target="_blank" style="color: #007bff;">Cohere API (command-r-plus-08-2024)</a>
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
                    chat_fn = partial(chat_with_character_stream, character_maker=character_maker) #
                    gr.ChatInterface(
                        chat_fn,
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
                        Logger.save_chat_log,
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
                    log_file_dropdown = gr.Dropdown(label="ログファイル選択", choices=Logger.list_log_files())
                    refresh_log_list_button = gr.Button("ログファイルリストを更新")
                    resume_chat_button = gr.Button("選択したログから会話を再開")

                    def update_log_dropdown() -> gr.update:
                        """ログファイルのドロップダウンを更新します"""
                        return gr.update(choices=Logger.list_log_files())

                    def load_and_display_chat_log(file_name: str) -> gr.update:
                        """チャットログを読み込んで表示します"""
                        chat_history = Logger.load_chat_log(file_name)
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

                    def resume_chat_and_switch_tab(chat_history: List[List[str]],character_maker: CharacterMaker) -> Tuple[gr.update, gr.update]:
                        """ログからチャットを再開し、タブを切り替えます"""
                        chatbot_ui = resume_chat_from_log(chat_history, character_maker)
                        return chatbot_ui, gr.update(selected="chat_tab")

                    resume_chat_button.click(
                        partial(resume_chat_and_switch_tab, character_maker=character_maker),
                        inputs=[chatbot_read],
                        outputs=[chatbot, tabs]
                    )

                with gr.Tab("設定"):
                    output = gr.Textbox(label="更新状態")

                    config = Config(DEFAULT_INI_FILE)

                    def update_dropdown(current_value):
                        model_files = model_manager.get_model_files(MODEL_DIR)
                        if "Cohere API (command-r-plus-08-2024)" not in model_files:
                            model_files.append("Cohere API (command-r-plus-08-2024)")

                        if current_value not in model_files:
                            model_files.insert(0, current_value)
                            status = f"現在のモデル（{current_value}）が見つかりません。ダウンロードしてください。"
                        else:
                            status = "モデルリストを更新しました。"
                        return gr.update(choices=model_files, value=current_value), status

                    with gr.Column():
                        gr.Markdown("### モデル設定")
                        model_settings = []

                        for key in ['DEFAULT_CHAT_MODEL', 'DEFAULT_GEN_MODEL']:
                            if key in config.config['Models']:
                                model_files = model_manager.get_model_files(MODEL_DIR)
                                if "Cohere API (command-r-plus-08-2024)" not in model_files:
                                    model_files.insert(0, "Cohere API (command-r-plus-08-2024)")

                                with gr.Row():
                                    dropdown = gr.Dropdown(
                                        label=key,
                                        choices=model_files,
                                        value=config.config['Models'].get(key, "Cohere API (command-r-plus-08-2024)")
                                    )
                                    refresh_button = gr.Button("更新", size="sm")
                                    status_message = gr.Markdown()

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

                        gr.Markdown("### Cohere API設定")

                        cohere_api_key = gr.Textbox(
                            label="Cohere API Key",
                            type="password",
                            value=config.get_cohere_api_key()
                        )

                        def save_cohere_api_key(api_key: str) -> str:
                            config.set_cohere_api_key(api_key)
                            return "Cohere API Keyが保存されました。"

                        save_api_key_button = gr.Button("Cohere API Keyを保存")
                        save_api_key_button.click(
                            save_cohere_api_key,
                            inputs=[cohere_api_key],
                            outputs=[output]
                        )

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
                        apply_fn = partial(apply_settings, character_maker)
                        apply_ini_settings_button.click(
                            apply_fn,
                            outputs=[output]
                        )

        return iface

async def start_gradio() -> None:
    """Gradioアプリケーションを起動します"""

    if not os.path.exists(DEFAULT_INI_FILE):
        print(f"{DEFAULT_INI_FILE} が見つかりません。デフォルト設定で作成します。")
        Settings.create_default_ini(DEFAULT_INI_FILE)

    config = Config(DEFAULT_INI_FILE)
    settings = Settings._parse_config(config.config)

    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    model_manager = ModelManager(MODEL_DIR)
    
    # CharacterMakerを初期化します
    character_maker = CharacterMaker(None, settings, model_dir=MODEL_DIR)
    
    # Cohere APIキーが設定されている場合は、Cohereアダプターを設定します
    if settings['DEFAULT_CHAT_MODEL'] == "Cohere API (command-r-plus-08-2024)":
        cohere_api_key = config.get_cohere_api_key()
        if cohere_api_key:
            character_maker.set_cohere_adapter(cohere_api_key)
        else:
            print("Cohere API Keyが設定されていません。Cohereモデルを使用する場合は、設定タブでAPIキーを入力してください。")

    demo = ChatApplicationGUI.build_gradio_interface(model_manager,character_maker)


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