import os
import json
import logging
from typing import Dict, List, Optional, Tuple
import azure.cognitiveservices.speech as speechsdk
import openai
import whisper
import torch
import torchaudio
from transformers import pipeline
import librosa
import soundfile as sf
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseSpeechRecognitionEngine:
    """음성인식 엔진의 기본 클래스"""
    
    def __init__(self, name: str):
        self.name = name
        self.config = Config()
    
    def transcribe(self, audio_path: str) -> str:
        """오디오 파일을 텍스트로 변환"""
        raise NotImplementedError
    
    def get_engine_info(self) -> Dict:
        """엔진 정보 반환"""
        return {
            "name": self.name,
            "type": self.__class__.__name__
        }

class AzureSpeechEngine(BaseSpeechRecognitionEngine):
    """Azure Speech Services 엔진"""
    
    def __init__(self):
        super().__init__("Azure Speech Services")
        if not self.config.AZURE_SPEECH_KEY:
            raise ValueError("Azure Speech Key가 설정되지 않았습니다.")
        
        # 작동하는 코드와 동일한 설정
        # 지역을 koreacentral로 설정 (작동하는 코드와 동일)
        azure_region = "koreacentral"
        
        self.speech_config = speechsdk.SpeechConfig(
            subscription=self.config.AZURE_SPEECH_KEY,
            region=azure_region
        )
        self.speech_config.speech_recognition_language = "ko-KR"
        
        logger.info(f"Azure Speech Services 초기화 완료 - 지역: {azure_region}")
        logger.info(f"Azure Key 길이: {len(self.config.AZURE_SPEECH_KEY)}")
        
    def transcribe(self, audio_path: str) -> str:
        try:
            # 작동하는 코드처럼 WAV 파일을 그대로 사용 (전처리 없음)
            if not audio_path.lower().endswith('.wav'):
                logger.warning("Azure는 WAV 파일만 지원합니다. 다른 형식은 변환 후 사용하세요.")
                return ""
            
            # 단순한 Azure 설정 (작동하는 코드와 동일)
            audio_config = speechsdk.AudioConfig(filename=audio_path)
            speech_recognizer = speechsdk.SpeechRecognizer(
                speech_config=self.speech_config, 
                audio_config=audio_config
            )
            
            # 음성 인식 실행 (단순하게)
            logger.info(f"Azure 음성인식 시작 - 파일: {os.path.basename(audio_path)}")
            result = speech_recognizer.recognize_once()
            
            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                recognized_text = result.text.strip().replace('.', '')
                logger.info(f"Azure 인식 성공: \"{recognized_text}\"")
                return recognized_text
            elif result.reason == speechsdk.ResultReason.NoMatch:
                logger.warning("Azure 인식 실패: 음성을 인식할 수 없음")
                return ""
            elif result.reason == speechsdk.ResultReason.Canceled:
                cancellation_details = result.cancellation_details
                logger.error(f"Azure 인식 취소됨: {cancellation_details.reason}")
                if cancellation_details.reason == speechsdk.CancellationReason.Error:
                    try:
                        error_details = cancellation_details.error_details
                        logger.error(f"Azure 오류 세부사항: {error_details}")
                    except AttributeError:
                        logger.error("Azure 오류 세부사항을 가져올 수 없습니다.")
                return ""
            else:
                logger.warning(f"Azure 인식 실패: {result.reason}")
                return ""
                
        except Exception as e:
            logger.error(f"Azure 음성인식 오류: {e}")
            return ""

class OpenAIWhisperEngine(BaseSpeechRecognitionEngine):
    """OpenAI Whisper API 엔진"""
    
    def __init__(self):
        super().__init__("OpenAI Whisper API")
        if not self.config.OPENAI_API_KEY:
            raise ValueError("OpenAI API Key가 설정되지 않았습니다.")
        
        openai.api_key = self.config.OPENAI_API_KEY
    
    def transcribe(self, audio_path: str) -> str:
        try:
            with open(audio_path, "rb") as audio_file:
                transcript = openai.Audio.transcribe(
                    "whisper-1",
                    audio_file,
                    language="ko"
                )
                return transcript["text"]
        except Exception as e:
            logger.error(f"OpenAI Whisper API 오류: {e}")
            return ""

class LocalWhisperEngine(BaseSpeechRecognitionEngine):
    """로컬 Whisper 모델 엔진"""
    
    def __init__(self, model_size: str = "base"):
        super().__init__(f"Local Whisper ({model_size})")
        self.model_size = model_size
        self.model = None
        self._load_model()
    
    def _load_model(self):
        try:
            import whisper
            self.model = whisper.load_model(self.model_size)
            logger.info(f"Whisper 모델 {self.model_size} 로드 완료")
        except ImportError:
            logger.warning("Whisper 모듈이 설치되지 않았습니다. pip install openai-whisper로 설치하세요.")
            self.model = None
        except Exception as e:
            logger.error(f"Whisper 모델 로드 실패: {e}")
            self.model = None
    
    def transcribe(self, audio_path: str) -> str:
        if not self.model:
            logger.warning("Whisper 모델이 로드되지 않았습니다.")
            return ""
        
        try:
            result = self.model.transcribe(audio_path, language="ko")
            return result["text"]
        except Exception as e:
            logger.error(f"로컬 Whisper 인식 오류: {e}")
            return ""

class GoogleSpeechEngine(BaseSpeechRecognitionEngine):
    """Google Cloud Speech-to-Text 엔진"""
    
    def __init__(self):
        super().__init__("Google Cloud Speech-to-Text")
        if not self.config.GOOGLE_APPLICATION_CREDENTIALS:
            raise ValueError("Google Cloud 인증 파일이 설정되지 않았습니다.")
        
        try:
            from google.cloud import speech
            self.client = speech.SpeechClient()
        except ImportError:
            logger.error("Google Cloud Speech 라이브러리가 설치되지 않았습니다.")
            self.client = None
    
    def transcribe(self, audio_path: str) -> str:
        if not self.client:
            return ""
        
        try:
            # 오디오 파일 읽기
            with open(audio_path, "rb") as audio_file:
                content = audio_file.read()
            
            audio = speech.RecognitionAudio(content=content)
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=self.config.SAMPLE_RATE,
                language_code="ko-KR",
                enable_automatic_punctuation=True,
                model="latest_long"  # 긴 오디오에 최적화
            )
            
            response = self.client.recognize(config=config, audio=audio)
            
            transcript = ""
            for result in response.results:
                transcript += result.alternatives[0].transcript + " "
            
            return transcript.strip()
            
        except Exception as e:
            logger.error(f"Google Speech 인식 오류: {e}")
            return ""

class Wav2VecEngine(BaseSpeechRecognitionEngine):
    """Facebook Wav2Vec2 모델 엔진 (한국어)"""
    
    def __init__(self):
        super().__init__("Wav2Vec2 (Korean)")
        self.pipe = None
        self._load_model()
    
    def _load_model(self):
        try:
            # 한국어 음성인식을 위한 Wav2Vec2 모델
            model_name = "kresnik/wav2vec2-large-xlsr-korean"
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=model_name,
                device=0 if torch.cuda.is_available() else -1
            )
            logger.info("Wav2Vec2 모델 로드 완료")
        except Exception as e:
            logger.error(f"Wav2Vec2 모델 로드 실패: {e}")
    
    def transcribe(self, audio_path: str) -> str:
        if not self.pipe:
            return ""
        
        try:
            # 오디오 전처리
            audio, sr = librosa.load(audio_path, sr=16000)
            
            # 모델 추론
            result = self.pipe(audio)
            return result["text"]
            
        except Exception as e:
            logger.error(f"Wav2Vec2 인식 오류: {e}")
            return ""

def get_all_engines() -> List[BaseSpeechRecognitionEngine]:
    """사용 가능한 모든 음성인식 엔진 반환 (무료 솔루션 우선)"""
    engines = []
    
    # 1. Azure Speech Services (월 5시간 무료)
    try:
        engines.append(AzureSpeechEngine())
        logger.info("Azure Speech Services 엔진 초기화 성공")
    except Exception as e:
        logger.warning(f"Azure 엔진 초기화 실패: {e}")
        logger.info("Azure Speech Key를 설정하거나 무료 할당량을 확인하세요")
    
    # 2. Local Whisper (완전 무료)
    try:
        engines.append(LocalWhisperEngine())
        logger.info("Local Whisper 엔진 초기화 성공")
    except Exception as e:
        logger.warning(f"로컬 Whisper 엔진 초기화 실패: {e}")
        logger.info("pip install openai-whisper로 설치하세요")
    
    # 3. Wav2Vec2 Korean (완전 무료)
    try:
        engines.append(Wav2VecEngine())
        logger.info("Wav2Vec2 Korean 엔진 초기화 성공")
    except Exception as e:
        logger.warning(f"Wav2Vec2 엔진 초기화 실패: {e}")
        logger.info("pip install transformers torch torchaudio로 설치하세요")
    
    # 4. OpenAI Whisper API (유료 - 선택사항)
    try:
        if os.getenv('OPENAI_API_KEY'):
            engines.append(OpenAIWhisperEngine())
            logger.info("OpenAI Whisper API 엔진 초기화 성공 (유료)")
        else:
            logger.info("ℹOpenAI Whisper API는 API 키가 필요합니다 (유료)")
    except Exception as e:
        logger.warning(f"OpenAI Whisper API 엔진 초기화 실패: {e}")
    
    # 5. Google Cloud Speech (유료 - 선택사항)
    try:
        if os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
            engines.append(GoogleSpeechEngine())
            logger.info("Google Cloud Speech 엔진 초기화 성공 (유료)")
        else:
            logger.info("Google Cloud Speech는 인증 파일이 필요합니다 (유료)")
    except Exception as e:
        logger.warning(f"Google Speech 엔진 초기화 실패: {e}")
    
    logger.info(f"총 {len(engines)}개의 엔진이 초기화되었습니다")
    
    if not engines:
        logger.error("사용 가능한 엔진이 없습니다. 설정을 확인하세요.")
    
    return engines 