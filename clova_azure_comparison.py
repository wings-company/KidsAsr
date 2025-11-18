#!/usr/bin/env python3
"""
Clova Speech와 Azure Speech Services 발화전사 비교 코드
wav/unrecognized 디렉토리의 파일들을 사용하여 두 서비스를 비교합니다.
"""

import os
import sys
import json
import time
import logging
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

# 프로젝트 모듈 import
from config import Config
from engine.modules.languagepoint_analyze_score import analyze_score

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('clova_azure_comparison.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class ClovaSpeechEngine:
    """Clova Speech API 엔진"""
    
    def __init__(self):
        self.config = Config()
        
        # invoke URL과 secret key 방식 우선 사용
        if self.config.CLOVA_INVOKE_URL and self.config.CLOVA_SECRET_KEY:
            self.url = self.config.CLOVA_INVOKE_URL
            self.secret_key = self.config.CLOVA_SECRET_KEY
            self.auth_type = 'secret_key'
            logger.info("Clova Speech API 초기화 완료 (invoke URL + secret key 방식)")
        # 기존 방식 (client ID + secret)도 지원
        elif self.config.CLOVA_CLIENT_ID and self.config.CLOVA_CLIENT_SECRET:
            self.client_id = self.config.CLOVA_CLIENT_ID
            self.client_secret = self.config.CLOVA_CLIENT_SECRET
            self.url = 'https://naveropenapi.apigw.ntruss.com/recog/v1/stt'
            self.auth_type = 'client_id'
            logger.info("Clova Speech API 초기화 완료 (client ID + secret 방식)")
        else:
            raise ValueError("Clova Speech API 키가 설정되지 않았습니다. CLOVA_INVOKE_URL과 CLOVA_SECRET_KEY 또는 CLOVA_CLIENT_ID와 CLOVA_CLIENT_SECRET을 설정하세요.")
    
    def transcribe(self, audio_path: str) -> str:
        """음성 인식 실행"""
        try:
            # invoke URL + secret key 방식
            if self.auth_type == 'secret_key':
                headers = {
                    'X-CLOVASPEECH-API-KEY': self.secret_key,
                    'Content-Type': 'application/octet-stream'
                }
                # URL에 lang 파라미터 추가 (한국어)
                url = self.url
                if '?' not in url:
                    url = f"{url}?lang=Kor"
                elif 'lang=' not in url:
                    url = f"{url}&lang=Kor"
            # client ID + secret 방식
            else:
                headers = {
                    'X-NCP-APIGW-API-KEY-ID': self.client_id,
                    'X-NCP-APIGW-API-KEY': self.client_secret,
                    'Content-Type': 'application/octet-stream'
                }
                # URL에 lang 파라미터 추가 (한국어)
                url = self.url
                if '?' not in url:
                    url = f"{url}?lang=Kor"
                elif 'lang=' not in url:
                    url = f"{url}&lang=Kor"
            
            with open(audio_path, 'rb') as audio_file:
                response = requests.post(url, headers=headers, data=audio_file, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                # 응답 형식에 따라 텍스트 추출
                recognized_text = result.get('text', '').strip()
                if not recognized_text:
                    # 다른 응답 형식 시도
                    recognized_text = result.get('result', {}).get('text', '').strip()
                logger.info(f"Clova 인식 성공: \"{recognized_text}\"")
                return recognized_text
            else:
                logger.error(f"Clova 인식 실패: {response.status_code}, {response.text}")
                return ""
                
        except Exception as e:
            logger.error(f"Clova 음성인식 오류: {e}")
            return ""


class AzureSpeechEngineWrapper:
    """Azure Speech Services 엔진 래퍼 (기존 코드 재사용)"""
    
    def __init__(self):
        import azure.cognitiveservices.speech as speechsdk
        self.speechsdk = speechsdk
        self.config = Config()
        
        if not self.config.AZURE_SPEECH_KEY:
            raise ValueError("Azure Speech Key가 설정되지 않았습니다.")
        
        azure_region = "koreacentral"
        self.speech_config = speechsdk.SpeechConfig(
            subscription=self.config.AZURE_SPEECH_KEY,
            region=azure_region
        )
        self.speech_config.speech_recognition_language = "ko-KR"
        
        logger.info(f"Azure Speech Services 초기화 완료 - 지역: {azure_region}")
    
    def transcribe(self, audio_path: str) -> str:
        """음성 인식 실행"""
        try:
            if not audio_path.lower().endswith('.wav'):
                logger.warning("Azure는 WAV 파일만 지원합니다.")
                return ""
            
            audio_config = self.speechsdk.AudioConfig(filename=audio_path)
            speech_recognizer = self.speechsdk.SpeechRecognizer(
                speech_config=self.speech_config,
                audio_config=audio_config
            )
            
            logger.info(f"Azure 음성인식 시작 - 파일: {os.path.basename(audio_path)}")
            result = speech_recognizer.recognize_once()
            
            if result.reason == self.speechsdk.ResultReason.RecognizedSpeech:
                recognized_text = result.text.strip().replace('.', '')
                logger.info(f"Azure 인식 성공: \"{recognized_text}\"")
                return recognized_text
            elif result.reason == self.speechsdk.ResultReason.NoMatch:
                logger.warning("Azure 인식 실패: 음성을 인식할 수 없음")
                return ""
            elif result.reason == self.speechsdk.ResultReason.Canceled:
                cancellation_details = result.cancellation_details
                logger.error(f"Azure 인식 취소됨: {cancellation_details.reason}")
                return ""
            else:
                logger.warning(f"Azure 인식 실패: {result.reason}")
                return ""
                
        except Exception as e:
            logger.error(f"Azure 음성인식 오류: {e}")
            return ""


class ClovaAzureComparator:
    """Clova Speech와 Azure Speech Services 비교 클래스"""
    
    def __init__(self):
        self.config = Config()
        self.clova_engine = None
        self.azure_engine = None
        self.results = []
        
        # 결과 저장 디렉토리
        self.output_dir = "results/clova_azure_comparison"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 엔진 초기화
        try:
            self.clova_engine = ClovaSpeechEngine()
        except Exception as e:
            logger.error(f"Clova 엔진 초기화 실패: {e}")
            logger.warning("CLOVA_CLIENT_ID와 CLOVA_CLIENT_SECRET을 .env 파일에 설정하세요.")
        
        try:
            self.azure_engine = AzureSpeechEngineWrapper()
        except Exception as e:
            logger.error(f"Azure 엔진 초기화 실패: {e}")
            logger.warning("AZURE_SPEECH_KEY를 .env 파일에 설정하세요.")
        
        if not self.clova_engine and not self.azure_engine:
            logger.error("사용 가능한 엔진이 없습니다.")
        
        logger.info("Clova-Azure 비교 초기화 완료")
    
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
    
    def test_single_file(self, audio_path: str, target_text: str) -> Dict:
        """단일 파일로 Clova와 Azure 테스트"""
        result = {
            'audio_file': Path(audio_path).name,
            'target_text': target_text,
            'clova_text': '',
            'azure_text': '',
            'clova_processing_time': 0.0,
            'azure_processing_time': 0.0,
        }
        
        # Clova 인식
        if self.clova_engine:
            try:
                start_time = time.time()
                clova_result = self.clova_engine.transcribe(audio_path)
                result['clova_processing_time'] = time.time() - start_time
                result['clova_text'] = clova_result
            except Exception as e:
                logger.error(f"Clova 인식 오류: {e}")
                result['clova_text'] = ''
        
        # Azure 인식
        if self.azure_engine:
            try:
                start_time = time.time()
                azure_result = self.azure_engine.transcribe(audio_path)
                result['azure_processing_time'] = time.time() - start_time
                result['azure_text'] = azure_result
            except Exception as e:
                logger.error(f"Azure 인식 오류: {e}")
                result['azure_text'] = ''
        
        # languagepoint_analyze_score를 사용한 분석
        if result['clova_text']:
            clova_analysis = analyze_score(target_text, result['clova_text'], use_optimal_matching=True)
            result.update({
                'clova_pccr_score': clova_analysis['PCC-R'][1] / clova_analysis['PCC-R'][0] if clova_analysis['PCC-R'][0] > 0 else 0.0,
                'clova_pcc_score': clova_analysis['PCC'][1] / clova_analysis['PCC'][0] if clova_analysis['PCC'][0] > 0 else 0.0,
                'clova_pvc_score': clova_analysis['PVC'][1] / clova_analysis['PVC'][0] if clova_analysis['PVC'][0] > 0 else 0.0,
                'clova_pwc_score': clova_analysis['PWC'][1],
                'clova_pmlu_score': clova_analysis['PMLU'][1] / clova_analysis['PMLU'][0] if clova_analysis['PMLU'][0] > 0 else 0.0,
            })
        else:
            result.update({
                'clova_pccr_score': 0.0,
                'clova_pcc_score': 0.0,
                'clova_pvc_score': 0.0,
                'clova_pwc_score': 0.0,
                'clova_pmlu_score': 0.0,
            })
        
        if result['azure_text']:
            azure_analysis = analyze_score(target_text, result['azure_text'], use_optimal_matching=True)
            result.update({
                'azure_pccr_score': azure_analysis['PCC-R'][1] / azure_analysis['PCC-R'][0] if azure_analysis['PCC-R'][0] > 0 else 0.0,
                'azure_pcc_score': azure_analysis['PCC'][1] / azure_analysis['PCC'][0] if azure_analysis['PCC'][0] > 0 else 0.0,
                'azure_pvc_score': azure_analysis['PVC'][1] / azure_analysis['PVC'][0] if azure_analysis['PVC'][0] > 0 else 0.0,
                'azure_pwc_score': azure_analysis['PWC'][1],
                'azure_pmlu_score': azure_analysis['PMLU'][1] / azure_analysis['PMLU'][0] if azure_analysis['PMLU'][0] > 0 else 0.0,
            })
        else:
            result.update({
                'azure_pccr_score': 0.0,
                'azure_pcc_score': 0.0,
                'azure_pvc_score': 0.0,
                'azure_pwc_score': 0.0,
                'azure_pmlu_score': 0.0,
            })
        
        # 비교 정보
        logger.info(f"{Path(audio_path).name}:")
        logger.info(f"  Clova: \"{result['clova_text']}\" (PCC-R={result['clova_pccr_score']:.3f}, PCC={result['clova_pcc_score']:.3f})")
        logger.info(f"  Azure: \"{result['azure_text']}\" (PCC-R={result['azure_pccr_score']:.3f}, PCC={result['azure_pcc_score']:.3f})")
        
        return result
    
    def run_comparison(self, data_dir: str):
        """Clova와 Azure 비교 실행"""
        logger.info("=== Clova Speech vs Azure Speech Services 비교 시작 ===")
        
        # 1. 테스트 데이터 로드
        test_data = self.load_test_data(data_dir)
        if not test_data:
            logger.error("테스트 데이터가 없습니다.")
            return
        
        if not self.clova_engine and not self.azure_engine:
            logger.error("사용 가능한 엔진이 없습니다.")
            return
        
        # 2. 각 파일로 테스트 실행
        all_results = []
        
        for test_item in test_data:
            logger.info(f"\n처리 중: {test_item['audio_file']}")
            result = self.test_single_file(
                test_item['audio_path'],
                test_item['target_text']
            )
            all_results.append(result)
        
        # 3. 결과 저장
        self.save_results(all_results)
        
        # 4. 결과 분석 및 비교
        self.analyze_and_compare(all_results)
        
        logger.info("=== Clova-Azure 비교 완료 ===")
    
    def save_results(self, results: List[Dict]):
        """결과 저장"""
        if not results:
            logger.warning("저장할 결과가 없습니다.")
            return
        
        # JSON으로 저장
        json_path = os.path.join(self.output_dir, 'clova_azure_comparison_results.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"결과가 {json_path}에 저장되었습니다.")
        
        # CSV로 저장
        df = pd.DataFrame(results)
        csv_path = os.path.join(self.output_dir, 'clova_azure_comparison_results.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        logger.info(f"결과가 {csv_path}에 저장되었습니다.")
        
        # 엑셀로 저장
        excel_path = os.path.join(self.output_dir, 'clova_azure_comparison_results.xlsx')
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Results', index=False)
            
            # 요약 정보도 별도 시트에 저장
            summary_data = []
            
            if 'clova_text' in df.columns:
                clova_success = (df['clova_text'].str.strip() != '').sum()
                clova_total = len(df)
                summary_data.append({
                    '서비스': 'Clova Speech',
                    '성공': clova_success,
                    '전체': clova_total,
                    '성공률': f"{clova_success/clova_total*100:.1f}%" if clova_total > 0 else "0%",
                    '평균 PCC-R': df['clova_pccr_score'].mean() if 'clova_pccr_score' in df.columns else 0.0,
                    '평균 PCC': df['clova_pcc_score'].mean() if 'clova_pcc_score' in df.columns else 0.0,
                    '평균 PVC': df['clova_pvc_score'].mean() if 'clova_pvc_score' in df.columns else 0.0,
                    '평균 처리시간': df['clova_processing_time'].mean() if 'clova_processing_time' in df.columns else 0.0,
                })
            
            if 'azure_text' in df.columns:
                azure_success = (df['azure_text'].str.strip() != '').sum()
                azure_total = len(df)
                summary_data.append({
                    '서비스': 'Azure Speech',
                    '성공': azure_success,
                    '전체': azure_total,
                    '성공률': f"{azure_success/azure_total*100:.1f}%" if azure_total > 0 else "0%",
                    '평균 PCC-R': df['azure_pccr_score'].mean() if 'azure_pccr_score' in df.columns else 0.0,
                    '평균 PCC': df['azure_pcc_score'].mean() if 'azure_pcc_score' in df.columns else 0.0,
                    '평균 PVC': df['azure_pvc_score'].mean() if 'azure_pvc_score' in df.columns else 0.0,
                    '평균 처리시간': df['azure_processing_time'].mean() if 'azure_processing_time' in df.columns else 0.0,
                })
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        logger.info(f"엑셀 결과가 {excel_path}에 저장되었습니다.")
    
    def analyze_and_compare(self, results: List[Dict]):
        """결과 분석 및 비교"""
        if not results:
            return
        
        df = pd.DataFrame(results)
        
        print("\n" + "="*60)
        print("Clova Speech vs Azure Speech Services 비교 결과")
        print("="*60)
        
        # 성공률 비교
        if 'clova_text' in df.columns and 'azure_text' in df.columns:
            clova_success = (df['clova_text'].str.strip() != '').sum()
            azure_success = (df['azure_text'].str.strip() != '').sum()
            total = len(df)
            
            print(f"\n전체 오디오 파일 수: {total}개")
            print(f"\n인식 성공률:")
            print(f"  Clova Speech: {clova_success}개 / {total}개 ({clova_success/total*100:.1f}%)")
            print(f"  Azure Speech: {azure_success}개 / {total}개 ({azure_success/total*100:.1f}%)")
            
            # 평균 성능 비교
            print(f"\n평균 성능:")
            if 'clova_pccr_score' in df.columns:
                print(f"  Clova PCC-R: {df['clova_pccr_score'].mean():.3f} ± {df['clova_pccr_score'].std():.3f}")
                print(f"  Clova PCC: {df['clova_pcc_score'].mean():.3f} ± {df['clova_pcc_score'].std():.3f}")
                print(f"  Clova PVC: {df['clova_pvc_score'].mean():.3f} ± {df['clova_pvc_score'].std():.3f}")
            
            if 'azure_pccr_score' in df.columns:
                print(f"  Azure PCC-R: {df['azure_pccr_score'].mean():.3f} ± {df['azure_pccr_score'].std():.3f}")
                print(f"  Azure PCC: {df['azure_pcc_score'].mean():.3f} ± {df['azure_pcc_score'].std():.3f}")
                print(f"  Azure PVC: {df['azure_pvc_score'].mean():.3f} ± {df['azure_pvc_score'].std():.3f}")
            
            # 처리 시간 비교
            if 'clova_processing_time' in df.columns and 'azure_processing_time' in df.columns:
                print(f"\n평균 처리 시간:")
                print(f"  Clova: {df['clova_processing_time'].mean():.2f}초")
                print(f"  Azure: {df['azure_processing_time'].mean():.2f}초")
            
            # 승자 결정
            print(f"\n=== 승자 ===")
            if clova_success > azure_success:
                print(f"  인식 성공률: Clova Speech 승리 ({clova_success} vs {azure_success})")
            elif azure_success > clova_success:
                print(f"  인식 성공률: Azure Speech 승리 ({azure_success} vs {clova_success})")
            else:
                print(f"  인식 성공률: 동점 ({clova_success} vs {azure_success})")
            
            if 'clova_pccr_score' in df.columns and 'azure_pccr_score' in df.columns:
                if df['clova_pccr_score'].mean() > df['azure_pccr_score'].mean():
                    print(f"  PCC-R 점수: Clova Speech 승리 ({df['clova_pccr_score'].mean():.3f} vs {df['azure_pccr_score'].mean():.3f})")
                elif df['azure_pccr_score'].mean() > df['clova_pccr_score'].mean():
                    print(f"  PCC-R 점수: Azure Speech 승리 ({df['azure_pccr_score'].mean():.3f} vs {df['clova_pccr_score'].mean():.3f})")
                else:
                    print(f"  PCC-R 점수: 동점 ({df['clova_pccr_score'].mean():.3f} vs {df['azure_pccr_score'].mean():.3f})")
        
        print("="*60)


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Clova Speech vs Azure Speech Services 비교')
    parser.add_argument('--data_dir', type=str, default='wav/unrecognized',
                       help='테스트 데이터 디렉토리 (기본값: wav/unrecognized)')
    
    args = parser.parse_args()
    
    # 데이터 디렉토리 확인
    if not os.path.exists(args.data_dir):
        print(f"데이터 디렉토리 {args.data_dir}가 존재하지 않습니다.")
        return
    
    # 비교 실행
    comparator = ClovaAzureComparator()
    comparator.run_comparison(args.data_dir)


if __name__ == "__main__":
    main()

