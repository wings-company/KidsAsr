"""
유틸리티 함수들
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple
import librosa
import soundfile as sf
from pathlib import Path

logger = logging.getLogger(__name__)

def validate_audio_file(audio_path: str) -> bool:
    """오디오 파일 유효성 검사"""
    try:
        # 파일 존재 확인
        if not os.path.exists(audio_path):
            logger.error(f"오디오 파일이 존재하지 않습니다: {audio_path}")
            return False
        
        # 파일 크기 확인
        file_size = os.path.getsize(audio_path)
        if file_size == 0:
            logger.error(f"오디오 파일이 비어있습니다: {audio_path}")
            return False
        
        # 오디오 파일 로드 테스트
        audio, sr = librosa.load(audio_path, sr=None)
        if len(audio) == 0:
            logger.error(f"오디오 파일을 로드할 수 없습니다: {audio_path}")
            return False
        
        logger.info(f"오디오 파일 검증 성공: {audio_path} (길이: {len(audio)/sr:.2f}초, 샘플레이트: {sr}Hz)")
        return True
        
    except Exception as e:
        logger.error(f"오디오 파일 검증 실패 ({audio_path}): {e}")
        return False

def convert_audio_format(input_path: str, output_path: str, target_sr: int = 16000, 
                        target_format: str = 'wav') -> bool:
    """오디오 파일 형식 변환"""
    try:
        # 오디오 로드
        audio, sr = librosa.load(input_path, sr=target_sr)
        
        # 형식 변환
        if target_format.lower() == 'wav':
            sf.write(output_path, audio, target_sr)
        else:
            # 다른 형식은 pydub 사용
            from pydub import AudioSegment
            audio_segment = AudioSegment(
                audio.tobytes(), 
                frame_rate=target_sr,
                sample_width=audio.dtype.itemsize, 
                channels=1
            )
            audio_segment.export(output_path, format=target_format)
        
        logger.info(f"오디오 변환 완료: {input_path} → {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"오디오 변환 실패: {e}")
        return False

def get_audio_info(audio_path: str) -> Dict:
    """오디오 파일 정보 추출"""
    try:
        audio, sr = librosa.load(audio_path, sr=None)
        
        # 기본 정보
        duration = len(audio) / sr
        channels = 1 if len(audio.shape) == 1 else audio.shape[1]
        
        # 스펙트럼 정보
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
        
        # MFCC (음성 특성)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        
        info = {
            'file_path': audio_path,
            'duration': duration,
            'sample_rate': sr,
            'channels': channels,
            'samples': len(audio),
            'spectral_centroid_mean': float(spectral_centroids.mean()),
            'spectral_rolloff_mean': float(spectral_rolloff.mean()),
            'mfcc_mean': float(mfccs.mean()),
            'mfcc_std': float(mfccs.std())
        }
        
        return info
        
    except Exception as e:
        logger.error(f"오디오 정보 추출 실패 ({audio_path}): {e}")
        return {}

def normalize_audio(audio_path: str, output_path: str, target_db: float = -20.0) -> bool:
    """오디오 정규화 (볼륨 레벨 조정)"""
    try:
        from pydub import AudioSegment
        from pydub.effects import normalize
        
        # 오디오 로드
        audio = AudioSegment.from_file(audio_path)
        
        # 정규화
        normalized_audio = normalize(audio)
        
        # 타겟 dB로 조정
        change_in_dB = target_db - normalized_audio.dBFS
        adjusted_audio = normalized_audio + change_in_dB
        
        # 저장
        adjusted_audio.export(output_path, format=output_path.split('.')[-1])
        
        logger.info(f"오디오 정규화 완료: {audio_path} → {output_path} (타겟: {target_db}dB)")
        return True
        
    except Exception as e:
        logger.error(f"오디오 정규화 실패: {e}")
        return False

def remove_silence(audio_path: str, output_path: str, 
                   silence_thresh: float = -40.0, 
                   min_silence_len: int = 500) -> bool:
    """무음 구간 제거"""
    try:
        from pydub import AudioSegment
        from pydub.silence import split_on_silence
        
        # 오디오 로드
        audio = AudioSegment.from_file(audio_path)
        
        # 무음 구간으로 분할
        audio_chunks = split_on_silence(
            audio, 
            min_silence_len=min_silence_len, 
            silence_thresh=silence_thresh
        )
        
        # 무음 구간을 제거하고 연결
        processed_audio = sum(audio_chunks)
        
        # 저장
        processed_audio.export(output_path, format=output_path.split('.')[-1])
        
        logger.info(f"무음 제거 완료: {audio_path} → {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"무음 제거 실패: {e}")
        return False

def create_audio_preview(audio_path: str, output_path: str, 
                        start_time: float = 0, duration: float = 10) -> bool:
    """오디오 미리보기 생성 (특정 구간 추출)"""
    try:
        from pydub import AudioSegment
        
        # 오디오 로드
        audio = AudioSegment.from_file(audio_path)
        
        # 구간 추출 (밀리초 단위)
        start_ms = int(start_time * 1000)
        end_ms = int((start_time + duration) * 1000)
        
        # 오디오 길이 확인
        if end_ms > len(audio):
            end_ms = len(audio)
        
        preview = audio[start_ms:end_ms]
        
        # 저장
        preview.export(output_path, format=output_path.split('.')[-1])
        
        logger.info(f"오디오 미리보기 생성 완료: {audio_path} → {output_path} ({start_time}s ~ {start_time + duration}s)")
        return True
        
    except Exception as e:
        logger.error(f"오디오 미리보기 생성 실패: {e}")
        return False

def batch_process_audio(input_dir: str, output_dir: str, 
                       process_func, **kwargs) -> List[str]:
    """배치 오디오 처리"""
    processed_files = []
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 지원하는 오디오 형식
    audio_extensions = {'.wav', '.mp3', '.m4a', '.flac', '.ogg'}
    
    for audio_file in Path(input_dir).glob('*'):
        if audio_file.suffix.lower() in audio_extensions:
            try:
                output_path = os.path.join(output_dir, audio_file.name)
                
                if process_func(str(audio_file), output_path, **kwargs):
                    processed_files.append(output_path)
                    logger.info(f"처리 완료: {audio_file.name}")
                else:
                    logger.warning(f"처리 실패: {audio_file.name}")
                    
            except Exception as e:
                logger.error(f"처리 중 오류 발생 ({audio_file.name}): {e}")
    
            total_files = len([f for f in Path(input_dir).glob('*') if f.suffix.lower() in audio_extensions])
        logger.info(f"배치 처리 완료: {len(processed_files)}/{total_files} 파일")
    return processed_files

def save_audio_metadata(audio_dir: str, output_file: str):
    """오디오 파일들의 메타데이터를 JSON으로 저장"""
    metadata = []
    
    audio_extensions = {'.wav', '.mp3', '.m4a', '.flac', '.ogg'}
    
    for audio_file in Path(audio_dir).glob('*'):
        if audio_file.suffix.lower() in audio_extensions:
            try:
                info = get_audio_info(str(audio_file))
                if info:
                    metadata.append(info)
            except Exception as e:
                logger.warning(f"메타데이터 추출 실패 ({audio_file.name}): {e}")
    
    # JSON으로 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    logger.info(f"메타데이터 저장 완료: {output_file} ({len(metadata)}개 파일)")
    return metadata 