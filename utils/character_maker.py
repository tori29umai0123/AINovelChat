import os
import threading
from typing import List, Dict, Any, Optional
from jinja2 import Template
from utils.llm_adapters import LlamaAdapter, CohereAdapter
from utils.settings import Settings

class CharacterMaker:
    def __init__(self, llm_adapter: Any, settings: Dict[str, Any], model_dir: str):
        self.llm_adapter = llm_adapter
        self.settings = settings
        self.model_dir = model_dir
        self.history: List[Dict[str, str]] = []
        self.chat_history: List[Dict[str, str]] = []
        self.use_chat_format = False
        self.model_loaded = threading.Event()
        self.current_model = None
        self.current_model_info = "モデルが選択されていません"
        self.model_lock = threading.Lock()
        self.chat_model_settings = None
        self.gen_model_settings = None
        self.use_cohere = False
        self.cohere_adapter = None
        self.cohere_api_key = None
        
    def set_cohere_adapter(self, api_key: str) -> None:
        self.cohere_adapter = CohereAdapter(api_key, self.settings)
        self.use_cohere = True
        self.current_model_info = "Cohere API (command-r-plus-08-2024)"
        print(f"現在のモデル: {self.current_model_info}")
        self.model_loaded.set()

    def update_cohere_settings(self) -> None:
        if self.use_cohere and self.cohere_adapter:
            self.cohere_adapter.update_settings(self.settings)

    def update_settings(self, new_settings: Dict[str, Any]) -> None:
        self.settings = new_settings
        self.chat_model_settings = None
        self.gen_model_settings = None
        self.update_cohere_settings()
        print("設定が更新されました。次回の生成時に新しい設定が適用されます。")

    def load_model(self, model_type: str) -> None:
        new_settings = self.get_current_settings(model_type)
        
        if model_type == 'CHAT':
            if self.chat_model_settings == new_settings:
                print(f"{model_type}モデルの設定に変更がないため、リロードをスキップします。")
                return
            self.chat_model_settings = new_settings
        else:
            if self.gen_model_settings == new_settings:
                print(f"{model_type}モデルの設定に変更がないため、リロードをスキップします。")
                return
            self.gen_model_settings = new_settings

        self.switch_model(model_type)

    def switch_model(self, model_type: str) -> None:
        new_settings = self.get_current_settings(model_type)
        new_model = new_settings['model_path']

        print(f"{model_type}モデルを{new_model}に切り替えます")

        if new_model == "Cohere API (command-r-plus-08-2024)":
            new_cohere_api_key = Settings.get_cohere_api_key(self.settings)
            if not self.use_cohere or (self.cohere_api_key != new_cohere_api_key):
                print("Cohereモデルに切り替えます")
                if new_cohere_api_key:
                    self.set_cohere_adapter(new_cohere_api_key)
                    self.cohere_api_key = new_cohere_api_key
                else:
                    print("Cohere API Keyが設定されていません。設定タブでAPIキーを入力してください。")
                    return
        else:
            if self.use_cohere:
                print("ローカルモデルに切り替えます")
                self.use_cohere = False
                self.cohere_adapter = None
            self.reload_model(model_type, new_settings)

        self.current_model = model_type
        print(f"現在のモデル: {self.current_model_info}")

    def reload_model(self, model_type: str, settings: Dict[str, Any]) -> None:
        self.model_loaded.clear()
        if self.use_cohere:
            print("Cohereモデルを使用中です。ローカルモデルのロードはスキップします。")
            self.model_loaded.set()
            return
        try:
            model_path = settings['model_path']
            full_model_path = os.path.join(self.model_dir, model_path)
            if not os.path.exists(full_model_path):
                raise FileNotFoundError(f"モデルファイルが見つかりません: {full_model_path}")
            
            llm_params = {k: v for k, v in settings.items() if k != 'model_path'}
            self.llm_adapter = LlamaAdapter(full_model_path, llm_params)
            self.current_model = model_type
            self.current_model_info = f"ローカルモデル: {model_path}"
            print(f"現在のモデル: {self.current_model_info}")
            self.model_loaded.set()
            print(f"{model_type}モデルをロードしました。モデルパス: {full_model_path}、GPUレイヤー数: {settings['n_gpu_layers']}")
        except FileNotFoundError as e:
            print(f"エラー: {str(e)}")
            print("設定タブでモデルパスを確認し、正しいモデルファイルをダウンロードしてください。")
            self.current_model_info = "モデルのロードに失敗しました"
        except Exception as e:
            print(f"{model_type}モデルのロード中にエラーが発生しました: {str(e)}")
            import traceback
            traceback.print_exc()
            self.current_model_info = "モデルのロードに失敗しました"
        finally:
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
        """チャットタブ用の応答を生成します（マルチターン会話）"""
        self.load_model('CHAT')  # 毎回モデルをロードし直す
        print(f"レスポンス生成中 - 現在のモデル: {self.current_model_info}")
        if self.use_cohere:
            return self.cohere_adapter.generate_response(input_str, self.chat_history)
        else:
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
        """生成タブ用のテキストを生成します（会話履歴なし）"""
        self.load_model('GEN')  # 毎回モデルをロードし直す
        print(f"テキスト生成中 - 現在のモデル: {self.current_model_info}")
        if self.use_cohere:
            return self.cohere_adapter.generate_text(text, instruction=instruction, gen_characters=gen_characters)
        else:
            if not self.model_loaded.wait(timeout=30) or not self.llm_adapter:
                return "モデルのロードに失敗しました。設定を確認してください。"
            
            author_description = self.settings.get('gen_author_description', '')
            max_tokens = int(gen_characters * gen_token_multiplier)
            
            try:
                if self.use_chat_format:
                    messages = [
                        {"role": "system", "content": author_description},
                        {"role": "user", "content": f"{instruction}\n\n{text}"}
                    ]
                    
                    response = self.llm_adapter.create_chat_completion(messages, max_tokens)
                    generated_text = response["choices"][0]["message"]["content"].strip()
                else:
                    prompt = f"{author_description}\n\n指示：{instruction}\n\n入力テキスト：{text}\n\n生成されたテキスト："
                    generated_text = self.llm_adapter.generate_text(prompt, max_tokens)
                
                if not generated_text:
                    print(f"生成されたテキストが空です。入力: {text[:100]}...")
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
        if self.use_cohere:
            self.cohere_adapter.reset_conversation()
