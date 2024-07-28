import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer, BertTokenizer, BertForSequenceClassification
import torchaudio

# 加载模型
speech_tokenizer = Wav2Vec2Tokenizer.from_pretrained('../models/speech_to_text_model')
speech_model = Wav2Vec2ForCTC.from_pretrained('../models/speech_to_text_model')
responsibility_tokenizer = BertTokenizer.from_pretrained('../models/responsibility_model')
responsibility_model = BertForSequenceClassification.from_pretrained('../models/responsibility_model')

# 语音转文本函数
def speech_to_text(audio_file):
    waveform, sample_rate = torchaudio.load(audio_file)
    inputs = speech_tokenizer(waveform.squeeze().numpy(), return_tensors='pt', sampling_rate=sample_rate)
    with torch.no_grad():
        logits = speech_model(inputs.input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = speech_tokenizer.decode(predicted_ids[0])
    return transcription

# 推理函数
def infer(audio_file):
    text = speech_to_text(audio_file)
    inputs = responsibility_tokenizer(text, return_tensors='pt')
    outputs = responsibility_model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    return "对方全责" if prediction == 1 else "自己全责"

# 示例调用
audio_file = '../data/audio_files/example.wav'
print(infer(audio_file))
