pip install git+https://github.com/WangHelin1997/SSR-Speech.git#subdirectory=audiocraft
pip install xformers==0.0.22
pip install openai-whisper>=20231117
pip install faster-whisper==1.0.1
pip install whisperx==3.3.2

pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
apt-get install ffmpeg
apt-get install espeak-ng
pip install tensorboard==2.16.2
pip install phonemizer==3.2.1
pip install datasets==2.16.0
pip install torchmetrics==0.11.1
pip install huggingface_hub==0.22.2

# only use for inference
pip install gradio==3.50.2
pip install nltk>=3.8.1
pip install num2words==0.5.13
pip install opencc-python-reimplemented
pip install ctranslate2==4.4.0