#!/usr/bin/env python3
# validator.py - 양자화 모델 검증

import os
import time
import json
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from .llama_cpp_python import LlamaCppInference

class ModelValidator:
    """양자화 모델 검증 클래스"""
    
    def __init__(self, output_dir="validation_results"):
        """
        검증 클래스 초기화
        
        Args:
            output_dir (str): 결과 저장 디렉토리
        """
        self.output_dir = output_dir
        
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
    
    def validate_models(
        self,
        original_model_path: str,
        quantized_model_path: str,
        validation_prompts: List[str],
        max_tokens: int = 256,
        n_ctx: int = 2048,
        n_gpu_layers: int = -1
    ) -> Dict[str, Any]:
        """원본 모델과 양자화 모델 비교 검증
        
        Args:
            original_model_path (str): 원본 모델 경로
            quantized_model_path (str): 양자화 모델 경로
            validation_prompts (list): 검증 프롬프트 목록
            max_tokens (int): 생성할 최대 토큰 수
            n_ctx (int): 컨텍스트 크기
            n_gpu_layers (int): GPU에 로드할 레이어 수
            
        Returns:
            dict: 검증 결과
        """
        # 결과 저장소
        results = {
            "original_model": original_model_path,
            "quantized_model": quantized_model_path,
            "samples": [],
            "metrics": {}
        }
        
        # 원본 모델 로드
        print(f"원본 모델 로드 중: {original_model_path}")
        try:
            original_model = LlamaCppInference(
                model_path=original_model_path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers
            )
        except Exception as e:
            print(f"원본 모델 로드 오류: {str(e)}")
            results["error"] = f"원본 모델 로드 실패: {str(e)}"
            return results
        
        # 양자화 모델 로드
        print(f"양자화 모델 로드 중: {quantized_model_path}")
        try:
            quantized_model = LlamaCppInference(
                model_path=quantized_model_path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers
            )
        except Exception as e:
            print(f"양자화 모델 로드 오류: {str(e)}")
            results["error"] = f"양자화 모델 로드 실패: {str(e)}"
            return results
        
        # 검증 샘플별 처리
        total_original_time = 0
        total_quantized_time = 0
        
        for i, prompt in enumerate(validation_prompts, 1):
            print(f"\n검증 샘플 {i}/{len(validation_prompts)}: {prompt[:50]}...")
            sample_result = {
                "prompt": prompt,
                "original": {},
                "quantized": {}
            }
            
            # 원본 모델 생성
            start_time = time.time()
            original_output = original_model.generate(prompt, max_tokens=max_tokens)
            original_time = time.time() - start_time
            total_original_time += original_time
            
            sample_result["original"] = {
                "output": original_output,
                "time": original_time
            }
            
            # 양자화 모델 생성
            start_time = time.time()
            quantized_output = quantized_model.generate(prompt, max_tokens=max_tokens)
            quantized_time = time.time() - start_time
            total_quantized_time += quantized_time
            
            sample_result["quantized"] = {
                "output": quantized_output,
                "time": quantized_time
            }
            
            # 품질 측정
            repetition_score_original = self._measure_repetition(original_output)
            repetition_score_quantized = self._measure_repetition(quantized_output)
            
            sample_result["metrics"] = {
                "speed_ratio": original_time / quantized_time if quantized_time > 0 else 0,
                "repetition_original": repetition_score_original,
                "repetition_quantized": repetition_score_quantized,
                "output_length_original": len(original_output),
                "output_length_quantized": len(quantized_output)
            }
            
            results["samples"].append(sample_result)
        
        # 종합 메트릭 계산
        avg_speed_ratio = sum(s["metrics"]["speed_ratio"] for s in results["samples"]) / len(results["samples"])
        avg_repetition_original = sum(s["metrics"]["repetition_original"] for s in results["samples"]) / len(results["samples"])
        avg_repetition_quantized = sum(s["metrics"]["repetition_quantized"] for s in results["samples"]) / len(results["samples"])
        
        results["metrics"] = {
            "avg_speed_ratio": avg_speed_ratio,
            "avg_repetition_original": avg_repetition_original,
            "avg_repetition_quantized": avg_repetition_quantized,
            "total_time_original": total_original_time,
            "total_time_quantized": total_quantized_time,
            "time_reduction_percent": (1 - (total_quantized_time / total_original_time)) * 100 if total_original_time > 0 else 0
        }
        
        # 결과 저장
        self._save_results(results)
        
        # 요약 출력
        print("\n검증 결과 요약:")
        print(f"속도 향상: {avg_speed_ratio:.2f}x (양자화 모델이 {results['metrics']['time_reduction_percent']:.1f}% 더 빠름)")
        print(f"반복 점수 (낮을수록 좋음) - 원본: {avg_repetition_original:.3f}, 양자화: {avg_repetition_quantized:.3f}")
        
        return results
    
    def _measure_repetition(self, text: str) -> float:
        """텍스트의 반복 패턴 정도 측정
        
        Args:
            text (str): 측정할 텍스트
            
        Returns:
            float: 반복 점수 (0~1, 높을수록 많은 반복)
        """
        if not text or len(text) < 20:
            return 0.0
        
        # 단어 수준 반복 측정
        words = text.split()
        if len(words) <= 1:
            return 0.0
            
        # 단어 n-gram 반복 계산 (2-gram, 3-gram, 4-gram)
        total_repetition_score = 0.0
        weights = [0.5, 0.3, 0.2]  # 2-gram, 3-gram, 4-gram 가중치
        
        for n, weight in zip([2, 3, 4], weights):
            if len(words) <= n:
                continue
                
            # n-gram 생성
            ngrams = [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
            
            # 중복 개수 계산
            unique_ngrams = set(ngrams)
            repetition_ratio = 1.0 - (len(unique_ngrams) / len(ngrams))
            
            total_repetition_score += repetition_ratio * weight
        
        # 문자 수준 반복 패턴 (반복 문자열) 감지
        char_repetition = 0.0
        for length in range(3, min(10, len(text) // 2)):
            for i in range(len(text) - length * 2):
                chunk1 = text[i:i+length]
                chunk2 = text[i+length:i+length*2]
                if chunk1 == chunk2:
                    char_repetition += length / len(text)
        
        # 총 반복 점수 (0~1 범위로 정규화)
        final_score = min(1.0, total_repetition_score + char_repetition)
        return final_score
    
    def _save_results(self, results: Dict[str, Any]) -> None:
        """검증 결과 저장
        
        Args:
            results (dict): 검증 결과
        """
        # 요약 파일 저장
        summary_path = os.path.join(self.output_dir, "validation_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            # 요약 결과만 저장 (샘플 제외)
            summary = {
                "original_model": results["original_model"],
                "quantized_model": results["quantized_model"],
                "metrics": results["metrics"],
                "num_samples": len(results["samples"])
            }
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # 상세 결과 저장
        details_path = os.path.join(self.output_dir, "validation_details.json")
        with open(details_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"검증 결과가 저장되었습니다: {summary_path}")
    
    def generate_optimal_settings(self, quantized_model_path: str) -> Dict[str, Any]:
        """양자화 모델의 최적 설정 생성
        
        Args:
            quantized_model_path (str): 양자화 모델 경로
            
        Returns:
            dict: 최적 설정 정보
        """
        # 기본 설정 정의
        base_settings = {
            "temperature": 0.8,
            "top_p": 0.9,
            "top_k": 40,
            "repeat_penalty": 1.1,
            "repeat_last_n": 64
        }
        
        # 다양한 설정 조합 정의
        setting_variations = [
            # 기본 설정
            base_settings.copy(),
            
            # 보수적 설정 (반복 방지 강화)
            {**base_settings, "repeat_penalty": 1.3, "repeat_last_n": 128},
            
            # 창의적 설정 (다양성 강화)
            {**base_settings, "temperature": 0.9, "top_p": 0.95, "repeat_penalty": 1.2},
            
            # Mirostat 설정
            {**base_settings, "mirostat_mode": 2, "mirostat_tau": 5.0, "mirostat_eta": 0.1},
            
            # 혼합 페널티 설정
            {**base_settings, "repeat_penalty": 1.2, "presence_penalty": 0.2, "frequency_penalty": 0.2}
        ]
        
        # 테스트 프롬프트
        test_prompts = [
            "인공지능의 미래에 대해 설명해주세요.",
            "아래 주제에 대한 에세이를 작성하세요: 지속 가능한 발전",
            "바다에 사는 생물들에 대해 상세히 알려주세요."
        ]
        
        try:
            # 모델 로드
            model = LlamaCppInference(
                model_path=quantized_model_path,
                n_ctx=2048,
                n_gpu_layers=-1
            )
            
            # 설정별 평가
            results = []
            
            for i, settings in enumerate(setting_variations, 1):
                setting_result = {
                    "settings": settings,
                    "samples": [],
                    "avg_repetition_score": 0.0
                }
                
                total_repetition = 0.0
                
                for prompt in test_prompts:
                    output = model.generate(prompt, max_tokens=256, **settings)
                    repetition_score = self._measure_repetition(output)
                    
                    setting_result["samples"].append({
                        "prompt": prompt,
                        "output": output,
                        "repetition_score": repetition_score
                    })
                    
                    total_repetition += repetition_score
                
                setting_result["avg_repetition_score"] = total_repetition / len(test_prompts)
                results.append(setting_result)
            
            # 최적 설정 선택 (반복 점수가 가장 낮은 것)
            results.sort(key=lambda x: x["avg_repetition_score"])
            optimal_settings = results[0]["settings"]
            
            # 결과 저장
            output_path = os.path.join(self.output_dir, "optimal_settings.json")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "model_path": quantized_model_path,
                    "optimal_settings": optimal_settings,
                    "all_tested_settings": [r["settings"] for r in results],
                    "repetition_scores": [r["avg_repetition_score"] for r in results]
                }, f, indent=2, ensure_ascii=False)
            
            print(f"최적 설정이 저장되었습니다: {output_path}")
            return optimal_settings
            
        except Exception as e:
            print(f"최적 설정 생성 오류: {str(e)}")
            return base_settings