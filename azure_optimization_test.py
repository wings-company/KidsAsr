#!/usr/bin/env python3
"""
Azure Speech Services 성능 향상을 위한 테스트 코드
wav/unrecognized 디렉토리의 파일들을 사용하여 다양한 Azure 설정을 테스트합니다.
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd

# 프로젝트 모듈 import
from config import Config
from engine.modules.languagepoint_analyze_score import analyze_score

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('azure_optimization_test.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class AzureOptimizationTester:
    """Azure Speech Services 최적화 테스트 클래스"""
    
    def __init__(self):
        self.config = Config()
        self.base_engine = None
        self.results = []
        
        # 결과 저장 디렉토리
        self.output_dir = "results/azure_optimization"
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info("Azure 최적화 테스트 초기화 완료")
    
    def load_test_data(self, data_dir: str) -> List[Dict]:
        """테스트 데이터 로드 (wav/unrecognized 디렉토리)"""
        test_data = []
        data_path = Path(data_dir)
        
        if not data_path.exists():
            logger.error(f"데이터 디렉토리가 존재하지 않습니다: {data_dir}")
            return test_data
        
        # .wav 파일 찾기
        for wav_file in data_path.glob('*.wav'):
            # 파일명에서 확장자 제거하여 정답 텍스트로 사용
            target_text = wav_file.stem
            
            test_data.append({
                'audio_path': str(wav_file),
                'audio_file': wav_file.name,
                'target_text': target_text
            })
            logger.info(f"테스트 데이터 로드: {wav_file.name} → 정답: {target_text}")
        
        logger.info(f"총 {len(test_data)}개의 테스트 데이터 로드 완료")
        return test_data
    
    def create_azure_engine_with_config(self, config_name: str, **kwargs) -> Optional[object]:
        """특정 설정으로 Azure 엔진 생성 (커스텀 클래스)"""
        try:
            import azure.cognitiveservices.speech as speechsdk
            
            if not self.config.AZURE_SPEECH_KEY:
                logger.error("Azure Speech Key가 설정되지 않았습니다.")
                return None
            
            # Azure 설정 생성 (기본 리전 사용)
            speech_config = speechsdk.SpeechConfig(
                subscription=self.config.AZURE_SPEECH_KEY,
                region='koreacentral'  # 기본 리전 사용
            )
            
            # 언어 설정
            language = kwargs.get('language', 'ko-KR')
            speech_config.speech_recognition_language = language
            
            # 추가 설정 적용
            if 'endpoint_id' in kwargs and kwargs['endpoint_id']:
                speech_config.endpoint_id = kwargs['endpoint_id']
            
            if 'profanity' in kwargs:
                if kwargs['profanity'] == 'Raw':
                    speech_config.set_profanity(speechsdk.ProfanityOption.Raw)
                elif kwargs['profanity'] == 'Removed':
                    speech_config.set_profanity(speechsdk.ProfanityOption.Removed)
                elif kwargs['profanity'] == 'Masked':
                    speech_config.set_profanity(speechsdk.ProfanityOption.Masked)
            
            if 'output_format' in kwargs:
                speech_config.output_format = kwargs['output_format']
            
            # 받아쓰기 모드 활성화 (발음 그대로 출력)
            try:
                speech_config.enable_dictation()
            except:
                pass  # 일부 SDK 버전에서는 지원하지 않을 수 있음
            
            # 단어 레벨 타임스탬프 요청 (더 정확한 인식)
            try:
                speech_config.request_word_level_timestamps()
            except:
                pass
            
            # 커스텀 엔진 객체 생성
            class CustomAzureEngine:
                def __init__(self, speech_config, config_name):
                    self.speech_config = speech_config
                    self.config_name = config_name
                    self.name = f"Azure ({config_name})"
                
                def transcribe(self, audio_path: str, preprocess: bool = True, max_retries: int = 3, use_voting: bool = True) -> str:
                    """음성 인식 실행 - 다중 투표 방식 포함"""
                    if use_voting:
                        # 혁신적 방법 1: 다중 투표 방식 (여러 번 인식해서 가장 많이 나온 결과 선택)
                        return self._transcribe_with_voting(audio_path, preprocess, num_votes=5)
                    else:
                        # 기존 재시도 로직
                        for attempt in range(max_retries):
                            try:
                                result = self._transcribe_once(audio_path, preprocess, attempt + 1)
                                if result:  # 성공하면 즉시 반환
                                    return result
                            except Exception as e:
                                logger.warning(f"인식 시도 {attempt + 1}/{max_retries} 실패: {e}")
                                if attempt < max_retries - 1:
                                    import time
                                    time.sleep(0.5)  # 재시도 전 잠시 대기
                        
                        # 모든 재시도 실패
                        logger.warning(f"모든 재시도 실패: {os.path.basename(audio_path)}")
                        return ""
                
                def _transcribe_with_voting(self, audio_path: str, preprocess: bool = True, num_votes: int = 5) -> str:
                    """다중 투표 방식: 여러 번 인식해서 가장 많이 나온 결과 선택"""
                    from collections import Counter
                    import time
                    
                    results = []
                    logger.info(f"다중 투표 방식 시작: {os.path.basename(audio_path)} ({num_votes}회 인식)")
                    
                    # 여러 번 인식
                    for vote_num in range(num_votes):
                        try:
                            result = self._transcribe_once(audio_path, preprocess, vote_num + 1)
                            if result and result.strip():  # 빈 문자열 제외
                                results.append(result.strip())
                                logger.debug(f"투표 {vote_num + 1}/{num_votes}: \"{result.strip()}\"")
                            time.sleep(0.2)  # 각 인식 사이 짧은 대기
                        except Exception as e:
                            logger.debug(f"투표 {vote_num + 1}/{num_votes} 실패: {e}")
                    
                    if not results:
                        logger.warning(f"다중 투표 실패: 모든 인식 결과가 비어있음")
                        return ""
                    
                    # 가장 많이 나온 결과 선택
                    result_counter = Counter(results)
                    most_common = result_counter.most_common(1)[0]
                    final_result = most_common[0]
                    count = most_common[1]
                    
                    logger.info(f"다중 투표 결과: \"{final_result}\" ({count}/{len(results)}회 일치, 총 {len(results)}개 결과)")
                    
                    # 과반수 이상이면 신뢰도 높음
                    if count >= len(results) // 2 + 1:
                        logger.info(f"높은 신뢰도: {count}/{len(results)}회 일치")
                    else:
                        # 낮은 신뢰도인 경우, 가장 긴 결과를 선택 (더 많은 정보 포함)
                        if len(result_counter) > 1:
                            longest_result = max(result_counter.keys(), key=len)
                            if len(longest_result) > len(final_result):
                                logger.info(f"낮은 신뢰도로 가장 긴 결과 선택: \"{longest_result}\" (기존: \"{final_result}\")")
                                final_result = longest_result
                        logger.warning(f"낮은 신뢰도: {count}/{len(results)}회 일치, 다른 결과들: {dict(result_counter)}")
                    
                    return final_result
                
                def _transcribe_once(self, audio_path: str, preprocess: bool = True, attempt: int = 1) -> str:
                    """음성 인식 실행 - 단일 시도"""
                    try:
                        if not audio_path.lower().endswith('.wav'):
                            logger.warning("Azure는 WAV 파일만 지원합니다.")
                            return ""
                        
                        # 오디오 전처리 (선택적)
                        final_audio_path = audio_path
                        if preprocess:
                            try:
                                import librosa
                                import soundfile as sf
                                import tempfile
                                import numpy as np
                                
                                # 오디오 로드 (원본 샘플레이트 유지)
                                audio, sr = librosa.load(audio_path, sr=None, mono=True)
                                
                                # 오디오 정보 로깅
                                duration = len(audio) / sr
                                max_amplitude = np.max(np.abs(audio))
                                logger.info(f"오디오 정보: {os.path.basename(audio_path)} - "
                                          f"길이: {duration:.2f}초, 샘플레이트: {sr}Hz, "
                                          f"최대볼륨: {max_amplitude:.4f}")
                                
                                # 오디오가 너무 조용하면 증폭
                                if max_amplitude < 0.1:
                                    gain = 0.5 / max_amplitude  # 최대 0.5까지 증폭
                                    audio = audio * min(gain, 10.0)  # 최대 10배까지만
                                    logger.info(f"오디오 볼륨 증폭: {min(gain, 10.0):.2f}배")
                                
                                # 볼륨 정규화 (아동 음성이 작을 수 있음)
                                audio = librosa.util.normalize(audio)
                                
                                # 샘플레이트를 16kHz로 변환 (Azure 권장)
                                if sr != 16000:
                                    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                                    sr = 16000
                                    logger.info(f"샘플레이트 변환: {sr}Hz")
                                
                                # 무음 구간 제거 (앞뒤)
                                # RMS 에너지 기반으로 무음 구간 찾기
                                frame_length = 2048
                                hop_length = 512
                                rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
                                rms_threshold = np.percentile(rms, 10)  # 하위 10%를 무음으로 간주
                                
                                # 앞뒤 무음 제거
                                frames_above_threshold = np.where(rms > rms_threshold)[0]
                                if len(frames_above_threshold) > 0:
                                    start_frame = max(0, frames_above_threshold[0] - 5)  # 여유 5프레임
                                    end_frame = min(len(rms), frames_above_threshold[-1] + 5)
                                    start_sample = start_frame * hop_length
                                    end_sample = end_frame * hop_length
                                    audio = audio[start_sample:end_sample]
                                    logger.info(f"무음 제거: {start_sample/sr:.2f}초 ~ {end_sample/sr:.2f}초")
                                
                                # 혁신적 방법 2: 짧은 오디오를 반복해서 길게 만들기 (Azure는 긴 오디오에서 더 잘 작동)
                                audio_duration = len(audio) / sr
                                if audio_duration < 2.0:  # 2초 미만이면 반복
                                    repeat_count = max(2, int(2.0 / audio_duration))  # 최소 2초가 되도록 반복
                                    repeat_count = min(repeat_count, 5)  # 최대 5회까지만 반복 (너무 길어지지 않도록)
                                    audio_repeated = np.tile(audio, repeat_count)
                                    audio = audio_repeated
                                    logger.info(f"오디오 반복: {audio_duration:.2f}초 → {len(audio)/sr:.2f}초 ({repeat_count}회 반복)")
                                
                                # 최소 길이 확인 (너무 짧으면 원본 사용)
                                if len(audio) / sr < 0.1:  # 0.1초 미만이면 너무 짧음
                                    logger.warning(f"오디오가 너무 짧음 ({len(audio)/sr:.2f}초), 원본 사용")
                                    final_audio_path = audio_path
                                else:
                                    # 임시 파일로 저장
                                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                                    sf.write(temp_file.name, audio, sr, subtype='PCM_16')
                                    final_audio_path = temp_file.name
                                    temp_file.close()
                                    
                                    logger.info(f"오디오 전처리 완료: {os.path.basename(audio_path)} "
                                              f"→ {len(audio)/sr:.2f}초")
                            except Exception as e:
                                logger.warning(f"오디오 전처리 실패, 원본 사용: {e}")
                                final_audio_path = audio_path
                        
                        audio_config = speechsdk.AudioConfig(filename=final_audio_path)
                        speech_recognizer = speechsdk.SpeechRecognizer(
                            speech_config=self.speech_config,
                            audio_config=audio_config
                        )
                        
                        logger.info(f"Azure 음성인식 시작 ({self.config_name}) - 파일: {os.path.basename(audio_path)} (시도 {attempt})")
                        
                        # 먼저 recognize_once()로 시도 (더 안정적)
                        result = speech_recognizer.recognize_once()
                        
                        # recognize_once() 결과 처리
                        recognized_text = ""
                        should_try_continuous = False
                        
                        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                            recognized_text = result.text.strip().replace('.', '')
                            
                            # 빈 문자열이면 실제로는 인식 실패이므로 연속 인식으로 재시도
                            if recognized_text:
                                logger.info(f"Azure 인식 성공 ({self.config_name}): \"{recognized_text}\"")
                                
                                # 임시 파일 정리
                                if final_audio_path != audio_path and os.path.exists(final_audio_path):
                                    try:
                                        os.unlink(final_audio_path)
                                    except:
                                        pass
                                
                                return recognized_text
                            else:
                                # 빈 문자열이면 실패로 처리하고 연속 인식으로 재시도
                                logger.debug(f"recognize_once()가 빈 문자열 반환, 연속 인식으로 재시도")
                                should_try_continuous = True
                        elif result.reason == speechsdk.ResultReason.NoMatch:
                            # NoMatch인 경우 연속 인식으로 재시도
                            logger.debug(f"recognize_once() 실패 (NoMatch), 연속 인식으로 재시도")
                            should_try_continuous = True
                        else:
                            # Canceled 등 다른 이유로 실패
                            logger.debug(f"recognize_once() 실패 ({result.reason}), 연속 인식으로 재시도")
                            should_try_continuous = True
                        
                        # 연속 인식으로 재시도
                        if should_try_continuous:
                            # 연속 인식 사용 (더 긴 오디오와 아동 음성에 적합)
                            all_results = []
                            done = False
                            
                            def recognized_cb(evt):
                                """인식 성공 이벤트"""
                                if evt.result.text:
                                    all_results.append(evt.result.text.strip())
                            
                            def canceled_cb(evt):
                                """취소 이벤트"""
                                nonlocal done
                                logger.debug(f"Azure 인식 취소됨 ({self.config_name}): {evt.reason}")
                                done = True
                            
                            def stop_cb(evt):
                                """중지 이벤트"""
                                nonlocal done
                                done = True
                            
                            # 이벤트 핸들러 등록
                            speech_recognizer.recognized.connect(recognized_cb)
                            speech_recognizer.canceled.connect(canceled_cb)
                            speech_recognizer.session_stopped.connect(stop_cb)
                            
                            # 연속 인식 시작
                            speech_recognizer.start_continuous_recognition()
                            
                            # 최대 30초 대기 (오디오 길이에 따라 조정)
                            import time
                            try:
                                import librosa
                                audio_duration = librosa.get_duration(path=final_audio_path)
                                timeout = max(audio_duration * 3, 5.0)  # 오디오 길이의 3배, 최소 5초
                                timeout = min(timeout, 30.0)  # 최대 30초
                            except:
                                timeout = 10.0
                            
                            start_time = time.time()
                            while not done and (time.time() - start_time) < timeout:
                                time.sleep(0.1)
                            
                            # 인식 중지
                            speech_recognizer.stop_continuous_recognition()
                            
                            # 결과 처리
                            if all_results:
                                recognized_text = ' '.join(all_results).strip().replace('.', '')
                                logger.info(f"Azure 인식 성공 ({self.config_name}, 연속인식): \"{recognized_text}\"")
                                
                                # 임시 파일 정리
                                if final_audio_path != audio_path and os.path.exists(final_audio_path):
                                    try:
                                        os.unlink(final_audio_path)
                                    except:
                                        pass
                                
                                return recognized_text
                            else:
                                # 임시 파일 정리
                                if final_audio_path != audio_path and os.path.exists(final_audio_path):
                                    try:
                                        os.unlink(final_audio_path)
                                    except:
                                        pass
                                
                                logger.warning(f"Azure 인식 실패 ({self.config_name}): 음성을 인식할 수 없음")
                                return ""  # 빈 문자열 반환하여 재시도 유도
                        else:
                            # Canceled 등 다른 이유로 실패
                            logger.debug(f"Azure 인식 실패 ({self.config_name}): {result.reason}")
                            
                            # 임시 파일 정리
                            if final_audio_path != audio_path and os.path.exists(final_audio_path):
                                try:
                                    os.unlink(final_audio_path)
                                except:
                                    pass
                            
                            return ""  # 빈 문자열 반환하여 재시도 유도
                            
                    except Exception as e:
                        logger.error(f"Azure 음성인식 오류 ({self.config_name}): {e}")
                        # 임시 파일 정리
                        if 'final_audio_path' in locals() and final_audio_path != audio_path and os.path.exists(final_audio_path):
                            try:
                                os.unlink(final_audio_path)
                            except:
                                pass
                        return ""
            
            engine = CustomAzureEngine(speech_config, config_name)
            logger.info(f"Azure 엔진 생성 완료 (설정: {config_name})")
            return engine
            
        except Exception as e:
            logger.error(f"Azure 엔진 생성 실패 ({config_name}): {e}")
            return None
    
    def test_azure_configuration(self, engine, audio_path: str, 
                                 target_text: str, config_name: str) -> Dict:
        """단일 Azure 설정으로 테스트"""
        start_time = time.time()
        
        try:
            # 음성 인식 실행
            hypothesis_text = engine.transcribe(audio_path)
            processing_time = time.time() - start_time
            
            # languagepoint_analyze_score를 사용한 분석
            analysis_result = analyze_score(target_text, hypothesis_text, use_optimal_matching=True)
            
            # 결과 구성
            result = {
                'config_name': config_name,
                'audio_file': Path(audio_path).name,
                'target_text': target_text,
                'hypothesis_text': hypothesis_text,
                'processing_time': processing_time,
                'pccr_opportunities': analysis_result['PCC-R'][0],
                'pccr_occurrences': analysis_result['PCC-R'][1],
                'pccr_score': analysis_result['PCC-R'][1] / analysis_result['PCC-R'][0] if analysis_result['PCC-R'][0] > 0 else 0.0,
                'pcc_opportunities': analysis_result['PCC'][0],
                'pcc_occurrences': analysis_result['PCC'][1],
                'pcc_score': analysis_result['PCC'][1] / analysis_result['PCC'][0] if analysis_result['PCC'][0] > 0 else 0.0,
                'pwc_score': analysis_result['PWC'][1],
                'pmlu_opportunities': analysis_result['PMLU'][0],
                'pmlu_occurrences': analysis_result['PMLU'][1],
                'pmlu_score': analysis_result['PMLU'][1] / analysis_result['PMLU'][0] if analysis_result['PMLU'][0] > 0 else 0.0,
                'pwp_score': analysis_result['PWP'][1],
                'pvc_opportunities': analysis_result['PVC'][0],
                'pvc_occurrences': analysis_result['PVC'][1],
                'pvc_score': analysis_result['PVC'][1] / analysis_result['PVC'][0] if analysis_result['PVC'][0] > 0 else 0.0,
            }
            
            logger.info(f"{config_name} - {Path(audio_path).name}: "
                       f"PCC-R={result['pccr_score']:.3f}, "
                       f"PCC={result['pcc_score']:.3f}, "
                       f"PVC={result['pvc_score']:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"{config_name} 테스트 실패 ({Path(audio_path).name}): {e}")
            return {
                'config_name': config_name,
                'audio_file': Path(audio_path).name,
                'target_text': target_text,
                'hypothesis_text': '',
                'processing_time': time.time() - start_time,
                'pccr_score': 0.0,
                'pcc_score': 0.0,
                'pwc_score': 0.0,
                'pmlu_score': 0.0,
                'pwp_score': 0.0,
                'pvc_score': 0.0,
                'error': str(e)
            }
    
    def get_test_configurations(self) -> List[Dict]:
        """테스트할 Azure 설정 목록"""
        configurations = [
            {
                'name': 'default',
                'description': '기본 설정 (ko-KR)',
                'params': {
                    'language': 'ko-KR'
                }
            },
            {
                'name': 'korean_enhanced',
                'description': '한국어 향상 설정 (Raw profanity)',
                'params': {
                    'language': 'ko-KR',
                    'profanity': 'Raw'  # 원문 그대로 출력
                }
            },
            # 추가 설정을 여기에 추가할 수 있습니다
            # 예: 커스텀 엔드포인트, 다른 언어 모델 등
        ]
        
        return configurations
    
    def run_optimization_test(self, data_dir: str):
        """최적화 테스트 실행"""
        logger.info("=== Azure Speech Services 최적화 테스트 시작 ===")
        
        # 1. 테스트 데이터 로드
        test_data = self.load_test_data(data_dir)
        if not test_data:
            logger.error("테스트 데이터가 없습니다.")
            return
        
        # 2. 테스트할 설정 목록 가져오기
        configurations = self.get_test_configurations()
        
        # 3. 각 설정으로 테스트 실행
        all_results = []
        
        for config in configurations:
            config_name = config['name']
            logger.info(f"\n설정 테스트 시작: {config_name} ({config['description']})")
            
            # Azure 엔진 생성
            engine = self.create_azure_engine_with_config(config_name, **config['params'])
            if not engine:
                logger.warning(f"{config_name} 엔진 생성 실패, 건너뜀")
                continue
            
            # 각 테스트 데이터로 테스트
            for test_item in test_data:
                result = self.test_azure_configuration(
                    engine, 
                    test_item['audio_path'],
                    test_item['target_text'],
                    config_name
                )
                all_results.append(result)
        
        # 4. 결과 저장
        self.save_results(all_results)
        
        # 5. 결과 분석 및 비교
        if all_results:
            self.analyze_results(all_results)
            
            # 성공/전체 갯수 통계 출력
            total_audio_count = len(test_data)
            total_test_count = len(all_results)
            
            # 설정별 성공률 계산
            print("\n" + "="*60)
            print("인식 성공률 통계")
            print("="*60)
            print(f"전체 오디오 파일 수: {total_audio_count}개")
            print(f"전체 테스트 수: {total_test_count}개 (설정 수 × 오디오 파일 수)")
            
            # 설정별로 성공률 계산
            for config in configurations:
                config_name = config['name']
                config_results = [r for r in all_results if r.get('config_name') == config_name]
                
                if config_results:
                    success_count = sum(1 for r in config_results if r.get('hypothesis_text', '').strip() != '')
                    total_count = len(config_results)
                    success_rate = (success_count / total_count * 100) if total_count > 0 else 0.0
                    
                    print(f"\n설정: {config_name} ({config['description']})")
                    print(f"  성공: {success_count}개 / 전체: {total_count}개 ({success_rate:.1f}%)")
            
            # 전체 성공률
            total_success = sum(1 for r in all_results if r.get('hypothesis_text', '').strip() != '')
            overall_success_rate = (total_success / total_test_count * 100) if total_test_count > 0 else 0.0
            print(f"\n전체 성공률: {total_success}개 / {total_test_count}개 ({overall_success_rate:.1f}%)")
            print("="*60)
        else:
            logger.warning("\n=== 테스트 결과가 없습니다 ===")
            logger.warning("Azure Speech Key가 설정되어 있는지 확인하세요.")
            logger.warning(".env 파일에 AZURE_SPEECH_KEY를 설정하거나 환경변수로 설정하세요.")
        
        logger.info("=== Azure 최적화 테스트 완료 ===")
    
    def save_results(self, results: List[Dict]):
        """결과 저장"""
        # 빈 결과 체크
        if not results:
            logger.warning("저장할 결과가 없습니다. Azure Speech Key가 설정되어 있는지 확인하세요.")
            # 빈 결과를 JSON으로 저장
            json_path = os.path.join(self.output_dir, 'azure_optimization_results.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump([], f, ensure_ascii=False, indent=2)
            logger.info(f"빈 결과가 {json_path}에 저장되었습니다.")
            return
        
        # JSON으로 저장
        json_path = os.path.join(self.output_dir, 'azure_optimization_results.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"결과가 {json_path}에 저장되었습니다.")
        
        # CSV로 저장
        df = pd.DataFrame(results)
        csv_path = os.path.join(self.output_dir, 'azure_optimization_results.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        logger.info(f"결과가 {csv_path}에 저장되었습니다.")
        
        # 엑셀로 저장
        excel_path = os.path.join(self.output_dir, 'azure_optimization_results.xlsx')
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Results', index=False)
            
            # 요약 정보도 별도 시트에 저장
            if 'config_name' in df.columns and len(df) > 0:
                summary_data = []
                for config_name in df['config_name'].unique():
                    config_df = df[df['config_name'] == config_name]
                    summary_data.append({
                        '설정명': config_name,
                        '평균 PCC-R': config_df['pccr_score'].mean() if 'pccr_score' in config_df.columns else 0.0,
                        '평균 PCC': config_df['pcc_score'].mean() if 'pcc_score' in config_df.columns else 0.0,
                        '평균 PVC': config_df['pvc_score'].mean() if 'pvc_score' in config_df.columns else 0.0,
                        '평균 PWC': config_df['pwc_score'].mean() if 'pwc_score' in config_df.columns else 0.0,
                        '평균 PMLU': config_df['pmlu_score'].mean() if 'pmlu_score' in config_df.columns else 0.0,
                        '평균 PWP': config_df['pwp_score'].mean() if 'pwp_score' in config_df.columns else 0.0,
                        '평균 처리시간': config_df['processing_time'].mean() if 'processing_time' in config_df.columns else 0.0,
                    })
                
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
            else:
                # 빈 요약 시트 생성
                empty_summary = pd.DataFrame([{'메시지': '결과가 없습니다. Azure Speech Key를 확인하세요.'}])
                empty_summary.to_excel(writer, sheet_name='Summary', index=False)
        
        logger.info(f"엑셀 결과가 {excel_path}에 저장되었습니다.")
    
    def analyze_results(self, results: List[Dict]):
        """결과 분석 및 비교"""
        if not results:
            return
        
        df = pd.DataFrame(results)
        
        logger.info("\n=== 결과 분석 ===")
        
        # 설정별 평균 성능
        for config_name in df['config_name'].unique():
            config_df = df[df['config_name'] == config_name]
            
            logger.info(f"\n설정: {config_name}")
            logger.info(f"  평균 PCC-R: {config_df['pccr_score'].mean():.3f} ± {config_df['pccr_score'].std():.3f}")
            logger.info(f"  평균 PCC: {config_df['pcc_score'].mean():.3f} ± {config_df['pcc_score'].std():.3f}")
            logger.info(f"  평균 PVC: {config_df['pvc_score'].mean():.3f} ± {config_df['pvc_score'].std():.3f}")
            logger.info(f"  평균 PWC: {config_df['pwc_score'].mean():.3f} ± {config_df['pwc_score'].std():.3f}")
            logger.info(f"  평균 PMLU: {config_df['pmlu_score'].mean():.3f} ± {config_df['pmlu_score'].std():.3f}")
            logger.info(f"  평균 처리시간: {config_df['processing_time'].mean():.2f}초")
        
        # 최고 성능 설정 찾기
        best_configs = {}
        for metric in ['pccr_score', 'pcc_score', 'pvc_score', 'pwc_score', 'pmlu_score']:
            if metric in df.columns:
                best_idx = df.groupby('config_name')[metric].mean().idxmax()
                best_value = df.groupby('config_name')[metric].mean().max()
                best_configs[metric] = (best_idx, best_value)
        
        logger.info("\n=== 최고 성능 설정 ===")
        for metric, (config_name, value) in best_configs.items():
            logger.info(f"  {metric}: {config_name} ({value:.3f})")


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Azure Speech Services 최적화 테스트')
    parser.add_argument('--data_dir', type=str, default='wav/unrecognized',
                       help='테스트 데이터 디렉토리 (기본값: wav/unrecognized)')
    
    args = parser.parse_args()
    
    # 데이터 디렉토리 확인
    if not os.path.exists(args.data_dir):
        print(f"데이터 디렉토리 {args.data_dir}가 존재하지 않습니다.")
        return
    
    # 테스트 실행
    tester = AzureOptimizationTester()
    tester.run_optimization_test(args.data_dir)


if __name__ == "__main__":
    main()

