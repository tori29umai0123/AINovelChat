import socket

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