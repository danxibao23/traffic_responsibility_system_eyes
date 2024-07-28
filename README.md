# traffic_responsibility_system
 用于小型事故判责，根据双方对于事故的描述进行一个简单的判责
 # 内容
  1、安装依赖<br>
  2、预处理语言模型<br>
  3、训练责任判定模型<br>
  4、推理从音频生成判定责任<br>
  5、Flask应用<br>
 # 文件结构
```markdown
 traffic_responsibility_system/
  ├── data/
  │   ├── accident_descriptions.csv
  │   └── audio_files/
  │       ├── example1.wav
  │       └── example2.wav
  ├── models/
  │   ├── speech_to_text_model/
  │   │   ├── config.json
  │   │   ├── pytorch_model.bin
  │   │   └── vocab.json
  │   └── responsibility_model/
  │       ├── config.json
  │       ├── pytorch_model.bin
  │       └── vocab.txt
  ├── scripts/
  │   ├── data_preprocessing.py
  │   ├── train_responsibility_model.py
  │   └── infer.py
  ├── app/
  │   └── app.py
  ├── requirements.txt
  └── README.md
```

 # 安装依赖
 
  包含项目的依赖
  ```markdown
    transformers
    torch
    torchaudio
    flask
    pandas
    scikit-learn
```
 # 预处理语言模型 scripts/data_preprocessing.py
  将语言模型转换为文本，并存储csv文件中
 
 # 训练责任判定模型 scripts/train_responsibility_model.py
   使用BERT模型训练责任判定模型
 #  推理 scripts/infer.py
  从音频文件生成责任判定结果
 #  Flask 应用 app/app.py
  创建一个简单的Flask应用，用于上传音频文件并获取责任判定结果
 # 运行步骤：
  1、在项目根目录下运行以下命令安装依赖：
    
    pip install -r requirements.txt
  2、预处理数据：
    
    python scripts/data_preprocessing.py
  3、训练责任判定模型：
    
    python scripts/train_responsibility_model.py
  4、启动Flask应用：
    
    python app/app.py
  5、打开浏览器访问 http://127.0.0.1:5000/upload，上传事故录音文件，查看判定结果。
# 注释
  数据示例<br>
  accident_descriptions.csv<br>
这个CSV文件包含了交通事故的描述以及责任判定的结果。例如：<br>
```markdown
 file,transcription,responsibility
 example1.wav,"I was driving on the right lane when the other car suddenly swerved into my lane.",0
 example2.wav,"The other driver ran a red light and hit my car from the side.",1
```
file：音频文件的名称。<br>
transcription：音频文件的文本转录内容。<br>
responsibility：责任判定的结果，0 表示自己全责，1 表示对方全责。<br>

udio_files/<br>
这个文件夹包含了所有的事故录音文件，例如：<br>
```markdown
audio_files/
├── example1.wav
├── example2.wav
```
训练和保存模型<br>
当你运行训练脚本时，模型会自动保存到 models/ 文件夹中。<br>
tips:需要自己有一定的数据集进行模型训练，数据的规模遇大，训练轮次遇多，模型的效果就遇高，数据集请按格式采集。
 
 
