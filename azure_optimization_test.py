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
                
                def transcribe(self, audio_path: str, preprocess: bool = True) -> str:
                    """음성 인식 실행 - 연속 인식 사용"""
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
                                
                                # 오디오 로드 및 정규화
                                audio, sr = librosa.load(audio_path, sr=16000, mono=True)
                                
                                # 볼륨 정규화 (아동 음성이 작을 수 있음)
                                audio = librosa.util.normalize(audio)
                                
                                # 임시 파일로 저장
                                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                                sf.write(temp_file.name, audio, sr)
                                final_audio_path = temp_file.name
                                temp_file.close()
                                
                                logger.debug(f"오디오 전처리 완료: {os.path.basename(audio_path)}")
                            except Exception as e:
                                logger.warning(f"오디오 전처리 실패, 원본 사용: {e}")
                                final_audio_path = audio_path
                        
                        audio_config = speechsdk.AudioConfig(filename=final_audio_path)
                        speech_recognizer = speechsdk.SpeechRecognizer(
                            speech_config=self.speech_config,
                            audio_config=audio_config
                        )
                        
                        logger.info(f"Azure 음성인식 시작 ({self.config_name}) - 파일: {os.path.basename(audio_path)}")
                        
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
                            logger.warning(f"Azure 인식 취소됨 ({self.config_name}): {evt.reason}")
                            done = True
                        
                        def stop_cb(evt):
                            """중지 이벤트"""
                            nonlocal done
                            done = True
                        
                        # 이벤트 핸들러 등록
                        speech_recognizer.recognized.connect(recognized_cb)
                        speech_recognizer.canceled.connect(canceled_cb)
                        speech_recognizer.session_stopped.connect(stop_cb)
                        speech_recognizer.session_started.connect(lambda evt: logger.debug(f"세션 시작: {evt}"))
                        
                        # 연속 인식 시작
                        speech_recognizer.start_continuous_recognition()
                        
                        # 최대 10초 대기 (오디오 길이에 따라 조정)
                        import time
                        timeout = 10.0
                        start_time = time.time()
                        while not done and (time.time() - start_time) < timeout:
                            time.sleep(0.1)
                        
                        # 인식 중지
                        speech_recognizer.stop_continuous_recognition()
                        
                        # 임시 파일 정리
                        if final_audio_path != audio_path and os.path.exists(final_audio_path):
                            try:
                                os.unlink(final_audio_path)
                            except:
                                pass
                        
                        # 결과 처리
                        if all_results:
                            recognized_text = ' '.join(all_results).strip().replace('.', '')
                            logger.info(f"Azure 인식 성공 ({self.config_name}): \"{recognized_text}\"")
                            return recognized_text
                        else:
                            # 연속 인식 실패 시 recognize_once() 재시도
                            logger.debug(f"연속 인식 결과 없음, recognize_once() 재시도")
                            result = speech_recognizer.recognize_once()
                            
                            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                                recognized_text = result.text.strip().replace('.', '')
                                logger.info(f"Azure 인식 성공 ({self.config_name}): \"{recognized_text}\"")
                                return recognized_text
                            elif result.reason == speechsdk.ResultReason.NoMatch:
                                logger.warning(f"Azure 인식 실패 ({self.config_name}): 음성을 인식할 수 없음")
                                return ""
                            else:
                                logger.warning(f"Azure 인식 실패 ({self.config_name}): {result.reason}")
                                return ""
                            
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

