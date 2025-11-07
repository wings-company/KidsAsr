import os

# .env 파일 자동 로드 (python-dotenv가 설치되어 있으면)
try:
    from dotenv import load_dotenv
    load_dotenv()
    print(".env 파일을 성공적으로 로드했습니다.")
except ImportError:
    print("python-dotenv가 설치되지 않았습니다. pip install python-dotenv로 설치하세요.")
    print("또는 환경변수를 직접 설정하세요.")

class Config:
    # Azure Speech Services
    AZURE_SPEECH_KEY = os.getenv('AZURE_SPEECH_KEY', '')
    AZURE_SPEECH_REGION = os.getenv('AZURE_SPEECH_REGION', 'eastus')
    
    # OpenAI Whisper API
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
    
    # Google Cloud Speech-to-Text
    GOOGLE_APPLICATION_CREDENTIALS = os.getenv('GOOGLE_APPLICATION_CREDENTIALS', '')
    
    # Audio settings
    SAMPLE_RATE = 16000
    CHANNELS = 1
    
    # Model settings
    WHISPER_MODEL = "base"  # tiny, base, small, medium, large
    
    # Evaluation metrics
    METRICS = ['wer', 'cer', 'accuracy']
    
    # Output paths
    OUTPUT_DIR = "results"
    AUDIO_DIR = "audio_samples"
    TRANSCRIPT_DIR = "transcripts" 