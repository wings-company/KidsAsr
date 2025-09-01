import os
import json
import logging
from typing import Dict, List, Tuple, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from jiwer import wer, cer
import re
from config import Config

# matplotlib 한국어 폰트 설정
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial Unicode MS', 'AppleGothic', 'Malgun Gothic']
plt.rcParams['axes.unicode_minus'] = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpeechRecognitionEvaluator:
    """음성인식 결과 평가 클래스"""
    
    def __init__(self):
        self.config = Config()
        self.metrics = {}
        
    def preprocess_text(self, text: str) -> str:
        """텍스트 전처리 (한국어 특화)"""
        if not text:
            return ""
        
        # 공백 정리
        text = re.sub(r'\s+', ' ', text.strip())
        
        # 특수문자 제거 (한국어는 유지)
        text = re.sub(r'[^\w\s가-힣]', '', text)
        
        # 소문자 변환 (한국어는 영향 없음)
        text = text.lower()
        
        return text
    
    def calculate_wer(self, reference: str, hypothesis: str) -> float:
        """Word Error Rate 계산"""
        if not reference or not hypothesis:
            return 1.0
        
        ref_words = self.preprocess_text(reference).split()
        hyp_words = self.preprocess_text(hypothesis).split()
        
        if not ref_words:
            return 1.0 if hyp_words else 0.0
        
        return wer(reference, hypothesis)
    
    def calculate_cer(self, reference: str, hypothesis: str) -> float:
        """Character Error Rate 계산 (한국어에 유용)"""
        if not reference or not hypothesis:
            return 1.0
        
        ref_chars = list(self.preprocess_text(reference))
        hyp_chars = list(self.preprocess_text(hypothesis))
        
        if not ref_chars:
            return 1.0 if hyp_chars else 0.0
        
        return cer(reference, hypothesis)
    
    def calculate_accuracy(self, reference: str, hypothesis: str) -> float:
        """정확도 계산 (1 - WER)"""
        wer_score = self.calculate_wer(reference, hypothesis)
        return 1.0 - wer_score
    
    def calculate_similarity(self, reference: str, hypothesis: str) -> float:
        """텍스트 유사도 계산 (Jaccard Similarity)"""
        if not reference or not hypothesis:
            return 0.0
        
        ref_words = set(self.preprocess_text(reference).split())
        hyp_words = set(self.preprocess_text(hypothesis).split())
        
        if not ref_words and not hyp_words:
            return 1.0
        
        intersection = len(ref_words.intersection(hyp_words))
        union = len(ref_words.union(hyp_words))
        
        return intersection / union if union > 0 else 0.0
    
    def evaluate_single_result(self, reference: str, hypothesis: str) -> Dict[str, float]:
        """단일 결과 평가"""
        return {
            'wer': self.calculate_wer(reference, hypothesis),
            'cer': self.calculate_cer(reference, hypothesis),
            'accuracy': self.calculate_accuracy(reference, hypothesis),
            'similarity': self.calculate_similarity(reference, hypothesis)
        }
    
    def evaluate_batch_results(self, results: List[Dict]) -> Dict[str, Any]:
        """배치 결과 평가"""
        if not results:
            return {}
        
        # 각 메트릭별로 통계 계산
        metrics_summary = {}
        for metric in ['wer', 'cer', 'accuracy', 'similarity']:
            values = [result[metric] for result in results if metric in result]
            if values:
                metrics_summary[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values)
                }
        
        return metrics_summary
    
    def compare_engines(self, engine_results: Dict[str, List[Dict]]) -> pd.DataFrame:
        """여러 엔진의 결과 비교"""
        comparison_data = []
        
        for engine_name, results in engine_results.items():
            if not results:
                continue
                
            # 각 메트릭별 평균 계산
            for result in results:
                row = {
                    'engine': engine_name,
                    'audio_file': result.get('audio_file', ''),
                    'wer': result.get('wer', 0),
                    'cer': result.get('cer', 0),
                    'accuracy': result.get('accuracy', 0),
                    'similarity': result.get('similarity', 0),
                    'processing_time': result.get('processing_time', 0)
                }
                comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        return df
    
    def generate_summary_report(self, comparison_df: pd.DataFrame) -> Dict[str, Any]:
        """요약 리포트 생성"""
        if comparison_df.empty:
            return {}
        
        # 엔진별 평균 성능
        engine_summary = comparison_df.groupby('engine').agg({
            'wer': ['mean', 'std'],
            'cer': ['mean', 'std'],
            'accuracy': ['mean', 'std'],
            'similarity': ['mean', 'std'],
            'processing_time': ['mean', 'std']
        }).round(4)
        
        # MultiIndex를 문자열 키로 변환
        engine_summary_dict = {}
        for engine_name in engine_summary.index:
            engine_summary_dict[engine_name] = {}
            for metric in ['wer', 'cer', 'accuracy', 'similarity', 'processing_time']:
                engine_summary_dict[engine_name][f'{metric}_mean'] = float(engine_summary.loc[engine_name, (metric, 'mean')])
                engine_summary_dict[engine_name][f'{metric}_std'] = float(engine_summary.loc[engine_name, (metric, 'std')])
        
        # 최고 성능 엔진 찾기
        best_wer = comparison_df.loc[comparison_df['wer'].idxmin()]
        best_cer = comparison_df.loc[comparison_df['cer'].idxmin()]
        best_accuracy = comparison_df.loc[comparison_df['accuracy'].idxmax()]
        best_similarity = comparison_df.loc[comparison_df['similarity'].idxmax()]
        
        summary = {
            'engine_summary': engine_summary_dict,
            'best_performance': {
                'lowest_wer': {
                    'engine': best_wer['engine'],
                    'audio_file': best_wer['audio_file'],
                    'value': float(best_wer['wer'])
                },
                'lowest_cer': {
                    'engine': best_cer['engine'],
                    'audio_file': best_cer['audio_file'],
                    'value': float(best_cer['cer'])
                },
                'highest_accuracy': {
                    'engine': best_accuracy['engine'],
                    'audio_file': best_accuracy['audio_file'],
                    'value': float(best_accuracy['accuracy'])
                },
                'highest_similarity': {
                    'engine': best_similarity['engine'],
                    'audio_file': best_similarity['audio_file'],
                    'value': float(best_similarity['similarity'])
                }
            },
            'total_samples': len(comparison_df),
            'unique_engines': comparison_df['engine'].nunique()
        }
        
        return summary
    
    def create_visualizations(self, comparison_df: pd.DataFrame, output_dir: str):
        """시각화 생성"""
        if comparison_df.empty:
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 숫자 컬럼만 선택 (audio_file 제외)
        numeric_cols = ['wer', 'cer', 'accuracy', 'similarity', 'processing_time']
        plot_df = comparison_df[['engine'] + numeric_cols].copy()
        
        # 1. WER 비교 차트
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        sns.boxplot(data=plot_df, x='engine', y='wer')
        plt.title('Word Error Rate 비교')
        plt.xticks(rotation=45)
        plt.ylabel('WER (낮을수록 좋음)')
        
        plt.subplot(2, 2, 2)
        sns.boxplot(data=plot_df, x='engine', y='cer')
        plt.title('Character Error Rate 비교')
        plt.xticks(rotation=45)
        plt.ylabel('CER (낮을수록 좋음)')
        
        plt.subplot(2, 2, 3)
        sns.boxplot(data=plot_df, x='engine', y='accuracy')
        plt.title('정확도 비교')
        plt.xticks(rotation=45)
        plt.ylabel('Accuracy (높을수록 좋음)')
        
        plt.subplot(2, 2, 4)
        sns.boxplot(data=plot_df, x='engine', y='processing_time')
        plt.title('처리 시간 비교')
        plt.xticks(rotation=45)
        plt.ylabel('Processing Time (초)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'performance_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 엔진별 평균 성능 막대 차트
        engine_avg = plot_df.groupby('engine')[numeric_cols].mean()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # WER
        axes[0, 0].bar(engine_avg.index, engine_avg['wer'])
        axes[0, 0].set_title('평균 WER')
        axes[0, 0].set_ylabel('WER')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # CER
        axes[0, 1].bar(engine_avg.index, engine_avg['cer'])
        axes[0, 1].set_title('평균 CER')
        axes[0, 1].set_ylabel('CER')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Accuracy
        axes[1, 0].bar(engine_avg.index, engine_avg['accuracy'])
        axes[1, 0].set_title('평균 정확도')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Similarity
        axes[1, 1].bar(engine_avg.index, engine_avg['similarity'])
        axes[1, 1].set_title('평균 유사도')
        axes[1, 1].set_ylabel('Similarity')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'average_performance.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. 상관관계 히트맵
        correlation_matrix = plot_df[numeric_cols].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, fmt='.3f')
        plt.title('메트릭 간 상관관계')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_results(self, comparison_df: pd.DataFrame, summary: Dict, output_dir: str):
        """결과 저장"""
        os.makedirs(output_dir, exist_ok=True)
        
        # CSV로 저장
        csv_path = os.path.join(output_dir, 'comparison_results.csv')
        comparison_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        logger.info(f"결과가 {csv_path}에 저장되었습니다.")
        
        # JSON으로 저장
        json_path = os.path.join(output_dir, 'evaluation_summary.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        logger.info(f"요약이 {json_path}에 저장되었습니다.")
        
        # 엑셀로 저장 (한국어 지원)
        excel_path = os.path.join(output_dir, 'comparison_results.xlsx')
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            comparison_df.to_excel(writer, sheet_name='Results', index=False)
            
            # 요약 정보도 별도 시트에 저장
            summary_df = pd.DataFrame([
                ['총 샘플 수', summary.get('total_samples', 0)],
                ['테스트된 엔진 수', summary.get('unique_engines', 0)],
                ['최고 정확도 엔진', summary.get('best_performance', {}).get('highest_accuracy', {}).get('engine', 'N/A')],
                ['최저 WER 엔진', summary.get('best_performance', {}).get('lowest_wer', {}).get('engine', 'N/A')]
            ], columns=['항목', '값'])
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        logger.info(f"엑셀 결과가 {excel_path}에 저장되었습니다.") 