# 必要ライブラリ
# pip install openai grpcio sounddevice numpy soundfile SpeechRecognition pyaudio

import grpc
import sounddevice as sd
import numpy as np
import openai
import speech_recognition as sr
from nvidia_riva import riva_tts_pb2, riva_tts_pb2_grpc  # Riva gRPCモジュール

# ----------------------------
# 設定
# ----------------------------
RIVA_SERVER = "localhost:50051"  # Rivaサーバー
LANGUAGE = "ja-JP"
VOICE = "alloy"
SAMPLE_RATE = 22050

openai.api_key = "YOUR_OPENAI_API_KEY"

# gRPCチャネル
channel = grpc.insecure_channel(RIVA_SERVER)
tts_client = riva_tts_pb2_grpc.RivaSpeechSynthesisStub(channel)

# 音声認識初期化
recognizer = sr.Recognizer()
mic = sr.Microphone()

# ----------------------------
# 音声再生関数
# ----------------------------
def speak(text):
    request = riva_tts_pb2.SynthesizeSpeechRequest(
        text=text,
        language_code=LANGUAGE,
        voice_name=VOICE,
        audio_format=riva_tts_pb2.AudioFormat.LINEAR16
    )
    response = tts_client.SynthesizeSpeech(request)
    audio_data = np.frombuffer(response.audio, dtype=np.int16)
    sd.play(audio_data, samplerate=SAMPLE_RATE)
    sd.wait()

# ----------------------------
# ChatGPT呼び出し関数
# ----------------------------
def chatgpt_response(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role":"user","content":prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

# ----------------------------
# メインループ（マイク入力）
# ----------------------------
print("マイク入力リアルタイム音声チャット開始（終了するには Ctrl+C）")
while True:
    try:
        with mic as source:
            print("話してください...")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)

        # 音声を文字起こし
        user_input = recognizer.recognize_google(audio, language="ja-JP")
        print("あなた:", user_input)

        # ChatGPTで応答生成
        reply_text = chatgpt_response(user_input)
        print("AI:", reply_text)

        # Rivaで音声出力
        speak(reply_text)

    except sr.UnknownValueError:
        print("音声を認識できませんでした。もう一度お願いします。")
    except sr.RequestError as e:
        print(f"音声認識サービスでエラー: {e}")
    except KeyboardInterrupt:
        print("\n終了します。")
        break

