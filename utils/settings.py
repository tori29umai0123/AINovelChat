import configparser
from typing import Dict, Any

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
        if 'Cohere' in config:
            settings['cohere_api_key'] = config['Cohere'].get('api_key', '')
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
        config['Cohere'] = {
            'api_key': settings.get('cohere_api_key', '')
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
            'DEFAULT_CHAT_MODEL': 'Cohere API (command-r-plus-08-2024)',
            'DEFAULT_GEN_MODEL': 'Cohere API (command-r-plus-08-2024)',
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
            'gen_n_ctx': 10000,
            'cohere_api_key': ''
        }
        Settings.save_to_ini(default_settings, filename)

    @staticmethod
    def load_from_ini(filename: str) -> Dict[str, Any]:
        """INIファイルから設定をロードします"""
        config = configparser.ConfigParser()
        config.read(filename, encoding='utf-8')
        return Settings._parse_config(config)

    @staticmethod
    def get_cohere_api_key(config: Dict[str, Any]) -> str:
        """Cohere APIキーを取得します"""
        return config.get('cohere_api_key', '')