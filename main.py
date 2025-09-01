#!/usr/bin/env python3
"""
아동 특화 음성인식 API 비교 프로젝트
Azure Speech Services와 다양한 오픈소스 솔루션들의 성능을 비교합니다.
"""

import os
import sys
import time
import logging
import argparse
from typing import Dict, List, Optional
import json
from pathlib import Path

# 프로젝트 모듈 import
from config import Config
from speech_recognition_engines import get_all_engines
from evaluation import SpeechRecognitionEvaluator

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('speech_comparison.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class SpeechComparisonRunner:
    """음성인식 비교 실행 클래스"""
    
    def __init__(self):
        self.config = Config()
        self.engines = get_all_engines()
        self.evaluator = SpeechRecognitionEvaluator()
        
        # 결과 저장 디렉토리 생성
        os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)
        os.makedirs(self.config.AUDIO_DIR, exist_ok=True)
        os.makedirs(self.config.TRANSCRIPT_DIR, exist_ok=True)
        
        logger.info(f"초기화된 엔진 수: {len(self.engines)}")
        for engine in self.engines:
            logger.info(f"  - {engine.name}")
    
    def load_test_data(self, data_dir: str) -> List[Dict]:
        """테스트 데이터 로드 (오디오 파일 + 정답 텍스트)"""
        test_data = []
        
        # 지원하는 오디오 형식
        audio_extensions = {'.wav', '.mp3', '.m4a', '.flac', '.ogg'}
        
        for audio_file in Path(data_dir).glob('*'):
            if audio_file.suffix.lower() in audio_extensions:
                # 정답 텍스트 파일 찾기 (같은 이름의 .txt 파일)
                transcript_file = audio_file.with_suffix('.txt')
                
                if transcript_file.exists():
                    with open(transcript_file, 'r', encoding='utf-8') as f:
                        reference_text = f.read().strip()
                    
                    test_data.append({
                        'audio_path': str(audio_file),
                        'audio_file': audio_file.name,
                        'reference_text': reference_text
                    })
                    logger.info(f"테스트 데이터 로드: {audio_file.name}")
                else:
                    logger.warning(f"정답 텍스트 파일을 찾을 수 없음: {audio_file.name}")
        
        logger.info(f"총 {len(test_data)}개의 테스트 데이터 로드 완료")
        return test_data
    
    def run_single_test(self, engine, audio_path: str, reference_text: str) -> Dict:
        """단일 테스트 실행"""
        start_time = time.time()
        
        try:
            # 음성인식 실행
            hypothesis_text = engine.transcribe(audio_path)
            processing_time = time.time() - start_time
            
            # 결과 평가
            metrics = self.evaluator.evaluate_single_result(reference_text, hypothesis_text)
            
            result = {
                'audio_file': Path(audio_path).name,
                'reference_text': reference_text,
                'hypothesis_text': hypothesis_text,
                'processing_time': processing_time,
                **metrics
            }
            
            logger.info(f"{engine.name}: {Path(audio_path).name} - WER: {metrics['wer']:.4f}, 시간: {processing_time:.2f}초")
            return result
            
        except Exception as e:
            logger.error(f"{engine.name} 테스트 실패 ({Path(audio_path).name}): {e}")
            return {
                'audio_file': Path(audio_path).name,
                'reference_text': reference_text,
                'hypothesis_text': '',
                'processing_time': time.time() - start_time,
                'wer': 1.0,
                'cer': 1.0,
                'accuracy': 0.0,
                'similarity': 0.0
            }
    
    def run_comparison(self, test_data: List[Dict]) -> Dict[str, List[Dict]]:
        """모든 엔진으로 비교 테스트 실행"""
        if not test_data:
            logger.error("테스트 데이터가 없습니다.")
            return {}
        
        if not self.engines:
            logger.error("사용 가능한 엔진이 없습니다.")
            return {}
        
        engine_results = {engine.name: [] for engine in self.engines}
        
        total_tests = len(test_data) * len(self.engines)
        current_test = 0
        
        logger.info(f"총 {total_tests}개의 테스트를 시작합니다...")
        
        for test_item in test_data:
            audio_path = test_item['audio_path']
            reference_text = test_item['reference_text']
            
            for engine in self.engines:
                current_test += 1
                logger.info(f"진행률: {current_test}/{total_tests} - {engine.name}: {test_item['audio_file']}")
                
                result = self.run_single_test(engine, audio_path, reference_text)
                engine_results[engine.name].append(result)
        
        logger.info("모든 테스트 완료!")
        return engine_results
    
    def save_detailed_results(self, engine_results: Dict[str, List[Dict]]):
        """상세 결과를 개별 파일로 저장"""
        for engine_name, results in engine_results.items():
            if not results:
                continue
            
            # 엔진별 결과를 JSON으로 저장
            output_file = os.path.join(self.config.TRANSCRIPT_DIR, f"{engine_name}_results.json")
            
            # 파일명에서 사용할 수 없는 문자 제거
            safe_filename = "".join(c for c in engine_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_filename = safe_filename.replace(' ', '_')
            output_file = os.path.join(self.config.TRANSCRIPT_DIR, f"{safe_filename}_results.json")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            logger.info(f"{engine_name} 결과가 {output_file}에 저장되었습니다.")
    
    def run_full_comparison(self, data_dir: str):
        """전체 비교 프로세스 실행"""
        logger.info("=== 아동 특화 음성인식 API 비교 시작 ===")
        
        # 1. 테스트 데이터 로드
        test_data = self.load_test_data(data_dir)
        if not test_data:
            logger.error(f"데이터 디렉토리 {data_dir}에서 테스트 데이터를 찾을 수 없습니다.")
            return
        
        # 2. 비교 테스트 실행
        engine_results = self.run_comparison(test_data)
        if not engine_results:
            logger.error("비교 테스트 실행에 실패했습니다.")
            return
        
        # 3. 결과 비교 및 평가
        comparison_df = self.evaluator.compare_engines(engine_results)
        summary = self.evaluator.generate_summary_report(comparison_df)
        
        # 4. 결과 저장
        self.save_detailed_results(engine_results)
        self.evaluator.save_results(comparison_df, summary, self.config.OUTPUT_DIR)
        
        # 5. 시각화 생성
        self.evaluator.create_visualizations(comparison_df, self.config.OUTPUT_DIR)
        
        # 6. 최종 요약 출력
        self.print_summary(summary, comparison_df)
        
        logger.info("=== 비교 완료! 결과는 results/ 디렉토리에 저장되었습니다. ===")
    
    def print_summary(self, summary: Dict, comparison_df):
        """결과 요약 출력"""
        print("\n" + "="*60)
        print("아동 음성인식 API 비교 결과 요약")
        print("="*60)
        
        if not summary:
            print("결과를 생성할 수 없습니다.")
            return
        
        print(f"\n전체 통계:")
        print(f"  • 총 테스트 샘플: {summary.get('total_samples', 0)}개")
        print(f"  • 테스트된 엔진: {summary.get('unique_engines', 0)}개")
        
        print(f"\n최고 성능:")
        best_perf = summary.get('best_performance', {})
        
        if 'lowest_wer' in best_perf:
            print(f"  • 최저 WER: {best_perf['lowest_wer']['engine']} ({best_perf['lowest_wer']['value']:.4f})")
        
        if 'highest_accuracy' in best_perf:
            print(f"  • 최고 정확도: {best_perf['highest_accuracy']['engine']} ({best_perf['highest_accuracy']['value']:.4f})")
        
        print(f"\n엔진별 평균 성능:")
        engine_summary = summary.get('engine_summary', {})
        
        for metric in ['wer', 'cer', 'accuracy']:
            if metric in engine_summary:
                print(f"\n  {metric.upper()}:")
                for engine_name in engine_summary[metric]['mean'].keys():
                    mean_val = engine_summary[metric]['mean'][engine_name]
                    std_val = engine_summary[metric]['std'][engine_name]
                    print(f"    • {engine_name}: {mean_val:.4f} ± {std_val:.4f}")
        
        print("\n" + "="*60)

# def create_sample_data():
#     """샘플 데이터 생성 (테스트용)"""
#     sample_dir = "sample_data"
#     os.makedirs(sample_dir, exist_ok=True)
    
#     # 샘플 정답 텍스트 파일들
#     sample_texts = [
#         "안녕하세요 저는 아이입니다",
#         "오늘 날씨가 좋네요",
#         "동물원에 가고 싶어요",
#         "엄마 아빠 사랑해요",
#         "학교에서 재미있게 놀았어요"
#     ]
    
#     for i, text in enumerate(sample_texts):
#         with open(os.path.join(sample_dir, f"sample_{i+1}.txt"), 'w', encoding='utf-8') as f:
#             f.write(text)
    
#     print(f"샘플 데이터가 {sample_dir}/ 디렉토리에 생성되었습니다.")
#     print("실제 오디오 파일을 이 디렉토리에 넣고 같은 이름의 .txt 파일과 함께 사용하세요.")

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='아동 특화 음성인식 API 비교')
    parser.add_argument('--data_dir', type=str, default='sample_data',
                       help='테스트 데이터 디렉토리 (기본값: sample_data)')
    parser.add_argument('--create_sample', action='store_true',
                       help='샘플 데이터 생성')
    parser.add_argument('--config', action='store_true',
                       help='설정 확인')
    
    args = parser.parse_args()
    
    if args.create_sample:
        create_sample_data()
        return
    
    if args.config:
        config = Config()
        print("현재 설정:")
        print(f"Azure Speech Key: {'설정됨' if config.AZURE_SPEECH_KEY else '설정되지 않음'}")
        print(f"OpenAI API Key: {'설정됨' if config.OPENAI_API_KEY else '설정되지 않음'}")
        print(f"Google Cloud Credentials: {'설정됨' if config.GOOGLE_APPLICATION_CREDENTIALS else '설정되지 않음'}")
        return
    
    # 데이터 디렉토리 확인
    if not os.path.exists(args.data_dir):
        print(f"데이터 디렉토리 {args.data_dir}가 존재하지 않습니다.")
        print("--create_sample 옵션으로 샘플 데이터를 생성하거나 올바른 디렉토리를 지정하세요.")
        return
    
    # 비교 실행
    runner = SpeechComparisonRunner()
    runner.run_full_comparison(args.data_dir)

if __name__ == "__main__":
    main() 