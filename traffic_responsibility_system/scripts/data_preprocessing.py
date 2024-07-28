import os
import pandas as pd
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer

# 加载预训练的Wav2Vec 2.0模型和分词器
tokenizer = Wav2Vec2Tokenizer.from_pretrained('facebook/wav2vec2-base-960h')
model = Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-base-960h')

# 读取语音文件并转换为文本
def speech_to_text(audio_file):
    waveform, sample_rate = torchaudio.load(audio_file)
    inputs = tokenizer(waveform.squeeze().numpy(), return_tensors='pt', sampling_rate=sample_rate)
    with torch.no_grad():
        logits = model(inputs.input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = tokenizer.decode(predicted_ids[0])
    return transcription

# 处理所有语音文件
audio_dir = '../data/audio_files/'
transcriptions = []
for audio_file in os.listdir(audio_dir):
    if audio_file.endswith('.wav'):
        transcript = speech_to_text(os.path.join(audio_dir, audio_file))
        transcriptions.append((audio_file, transcript))

# 保存到CSV文件
df = pd.DataFrame(transcriptions, columns=['file', 'transcription'])
df.to_csv('../data/accident_descriptions.csv', index=False)
