# AINovelChat
このアプリケーションは、AIキャラクターとチャットができる対話型アプリです。ユーザーはカスタマイズ可能なAIキャラクターと会話を楽しむことができます。

## ビルド方法
```
pyinstaller "AINovelChat.spec"
copy AINovelChat_model_DL.cmd dist\AINovelChat\AINovelChat_model_DL.cmd
copy custom.html dist\AINovelChat\custom.html
copy AINovelChat_ReadMe.txt dist\AINovelChat\AINovelChat_ReadMe.txt
```
