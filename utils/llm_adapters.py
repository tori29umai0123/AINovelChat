from typing import List, Dict, Any, Optional
from llama_cpp import Llama
import cohere

class LlamaAdapter:
    def __init__(self, model_path: str, params: Dict[str, Any]):
        self.model_path = model_path
        self.params = params
        params_copy = params.copy()
        if 'model_path' in params_copy:
            del params_copy['model_path']
        self.llm = Llama(model_path=model_path, **params_copy)

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
        try:
            response = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=self.params['temperature'],
                top_p=self.params['top_p'],
                top_k=self.params['top_k'],
                repeat_penalty=self.params['repeat_penalty'],
                stop=["user:", "・会話履歴", "<END>"]
            )
            print(f"LLM応答: {response}")
            return self._process_response(response)
        except Exception as e:
            print(f"LLM生成中にエラーが発生: {str(e)}")
            raise

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

class CohereAdapter:
    def __init__(self, api_key: str, settings: Dict[str, Any]):
        self.client = cohere.Client(api_key=api_key)
        self.chat_history: List[Dict[str, str]] = []
        self.api_key = api_key
        self.settings = settings

    def update_settings(self, new_settings: Dict[str, Any]):
        if 'cohere_api_key' in new_settings:
            self.api_key = new_settings['cohere_api_key']
        self.settings.update(new_settings)

    def generate_response(self, message: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        chat_system_message = f"{self.settings.get('chat_author_description', '')}\n\n{self.settings.get('chat_instructions', '')}"
        try:
            if not message.strip():
                return "メッセージが空です。有効なメッセージを入力してください。"
            
            self.chat_history = chat_history
            response = self.client.chat(
                model="command-r-plus-08-2024",
                chat_history=self.chat_history,
                message=message,
                preamble=chat_system_message
            )
            self.chat_history = response.chat_history
            return response.text
        except Exception as e:
            print(f"Cohere API エラー: {str(e)}")
            return f"エラーが発生しました: {str(e)}"

    def generate_text(self, message: str, instruction: str = "", gen_characters: str = "") -> str:
        characters_instruction = f"文字数は{gen_characters}以内にしてください"
        gen_system_message = f"{self.settings.get('gen_author_description', '')}\n\n{instruction}\n\n{characters_instruction}"
        try:
            full_message = f"{instruction}\n\n{message}".strip()
            if not full_message:
                return "指示とテキストが両方とも空です。少なくともどちらか一方を入力してください。"
            
            response = self.client.chat(
                model="command-r-plus-08-2024",
                conversation_id='user_defined_id_1',
                message=full_message,
                preamble=gen_system_message
            )
            return response.text
        except Exception as e:
            print(f"Cohere API エラー: {str(e)}")
            return f"エラーが発生しました: {str(e)}"

    def reset_conversation(self):
        self.chat_history = []

    def get_api_key(self):
        return self.api_key        
