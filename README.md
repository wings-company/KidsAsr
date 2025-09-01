# 아동 특화 음성인식 API 비교 프로젝트

다양한 오픈소스 음성인식 솔루션들의 아동 음성 인식 성능을 비교하는 프로젝트입니다.


## 설치

1. 저장소 클론
```bash
git clone <repository-url>
cd KidsAsr
```

2. 가상환경 생성 및 활성화
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. 의존성 설치
```bash
pip install -r requirements.txt
```

## 설정

### 환경 변수 설정

`.env` 파일을 생성하고 다음 정보를 입력하세요:

```bash
# Azure Speech Services
AZURE_SPEECH_KEY=your_azure_speech_key_here
AZURE_SPEECH_REGION=your_region

# OpenAI API
OPENAI_API_KEY=your_openai_api_key_here

# Google Cloud Speech-to-Text
GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/service-account-key.json
```

## 데이터 준비

### 테스트 데이터 구조

```
sample_data/
├── audio_1.wav
├── audio_1.txt
├── audio_2.mp3
├── audio_2.txt
└── ...
```

- **오디오 파일**: WAV, MP3, M4A, FLAC, OGG 형식 지원
- **정답 텍스트**: 같은 이름의 .txt 파일 (예: `audio_1.wav` → `audio_1.txt`)

### 샘플 데이터 생성

```bash
python main.py --create_sample
```

## 사용법

### 기본 실행

```bash
# 샘플 데이터로 테스트
python main.py

# 특정 디렉토리의 데이터로 테스트
python main.py --data_dir /path/to/your/audio/files

# 설정 확인
python main.py --config
```

### 실행 과정

1. **데이터 로드**: 지정된 디렉토리에서 오디오 파일과 정답 텍스트 로드
2. **엔진 초기화**: 사용 가능한 모든 음성인식 엔진 초기화
3. **음성인식 실행**: 각 엔진으로 모든 오디오 파일 처리
4. **성능 평가**: WER, CER, 정확도 등 계산
5. **결과 저장**: CSV, JSON, Excel 형식으로 저장
6. **시각화**: 성능 비교 차트 생성

## 결과 해석

### 평가 메트릭

- **WER (Word Error Rate)**: 단어 오류율 (낮을수록 좋음)
- **CER (Character Error Rate)**: 문자 오류율 (낮을수록 좋음)
- **Accuracy**: 정확도 (높을수록 좋음)
- **Similarity**: 텍스트 유사도 (높을수록 좋음)
- **Processing Time**: 처리 시간 (초)

### 결과 파일

- `results/comparison_results.csv`: 상세 비교 결과
- `results/evaluation_summary.json`: 요약 통계
- `results/comparison_results.xlsx`: Excel 형식 결과
- `results/performance_comparison.png`: 성능 비교 차트
- `transcripts/`: 각 엔진별 상세 결과

## 아동 음성 특화 기능

### Azure Speech Services 최적화
- 대화 모드 설정으로 자연스러운 음성 인식
- 한국어 특화 설정

### Wav2Vec2 한국어 모델
- 한국어 음성에 특화된 사전 훈련 모델
- 아동 음성 패턴 학습

### Whisper 모델
- 다양한 크기의 모델로 성능과 속도 조절
- 한국어 지원

## 주의사항

1. **API 비용**: Azure, OpenAI, Google Cloud API 사용 시 비용 발생
2. **음성 품질**: 오디오 파일의 품질이 인식 정확도에 큰 영향
3. **네트워크**: 클라우드 API 사용 시 안정적인 인터넷 연결 필요
4. **저장공간**: 로컬 모델 다운로드 시 추가 저장공간 필요

## 문제 해결

### 일반적인 오류

1. **API 키 오류**
   - `.env` 파일의 API 키 확인
   - API 키의 유효성 및 권한 확인

2. **오디오 파일 오류**
   - 지원되는 형식인지 확인
   - 파일 손상 여부 확인

3. **메모리 부족**
   - 로컬 모델 크기 조정
   - 배치 크기 줄이기

### 로그 확인

```bash
tail -f speech_comparison.log
```

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.
