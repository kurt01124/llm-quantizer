#!/usr/bin/env python3
# dynamic_quantizer.py - 통합 양자화 파이프라인

import os
import time
from typing import List, Dict, Any, Optional

from .model_loader import ModelLoader
from .architecture_analyzer import ArchitectureAnalyzer
from .importance_analyzer import ImportanceAnalyzer
from .quantization_planner import QuantizationPlanner
from .quantizer import Quantizer

class DynamicQuantizer:
    """LLM 동적 양자화 통합 클래스"""
    
    def __init__(self, model_name, output_dir="quantized_model", device="cuda"):
        """
        동적 양자화 클래스 초기화
        
        Args:
            model_name (str): 모델 이름
            output_dir (str): 결과 저장 디렉토리
            device (str): 사용할 디바이스
        """
        self.model_name = model_name
        self.output_dir = output_dir
        self.device = device
        
        # 하위 구성 요소 경로 설정
        self.architecture_dir = os.path.join(output_dir, "architecture_analysis")
        self.importance_dir = os.path.join(output_dir, "importance_analysis")
        self.quant_plan_dir = os.path.join(output_dir, "quantization_plan")
        self.quant_model_dir = os.path.join(output_dir, "quantized_model")
        
        # 상태 변수
        self.model = None
        self.tokenizer = None
        self.total_params = 0
        self.layer_sizes = {}
        self.moe_info = {}
        self.weight_stats = {}
        self.layer_importance = {}
        self.super_weights = []
        self.quantization_plan = {}
        
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"DynamicQuantizer 초기화: {model_name}")
        print(f"디바이스: {device}")
    
    def load_model(self):
        """모델 및 토크나이저 로드"""
        loader = ModelLoader(self.model_name, self.device)
        self.model, self.tokenizer = loader.load_model()
        return self.model
    
    def analyze_architecture(self):
        """모델 아키텍처 분석"""
        if self.model is None:
            self.load_model()
        
        analyzer = ArchitectureAnalyzer(self.model, self.architecture_dir)
        self.layer_sizes, self.total_params, self.moe_info = analyzer.analyze()
        
        return self.layer_sizes, self.total_params, self.moe_info
    
    def analyze_layer_importance(self, sample_texts=None, max_samples=3):
        """레이어별 중요도 분석"""
        if self.model is None:
            self.load_model()
        
        if not self.layer_sizes:
            self.analyze_architecture()
        
        analyzer = ImportanceAnalyzer(
            self.model, 
            self.tokenizer, 
            self.layer_sizes,
            self.importance_dir,
            self.device
        )
        
        self.weight_stats, self.layer_importance, self.super_weights = analyzer.analyze(
            sample_texts, 
            max_samples
        )
        
        return self.weight_stats, self.layer_importance, self.super_weights
    
    def design_quantization_strategy(self):
        """양자화 전략 설계"""
        if not self.layer_sizes:
            self.analyze_architecture()
        
        if not self.layer_importance:
            self.analyze_layer_importance()
        
        planner = QuantizationPlanner(
            self.model_name,
            self.layer_sizes,
            self.total_params,
            self.layer_importance,
            self.super_weights,
            self.quant_plan_dir,
            self.moe_info
        )
        
        self.quantization_plan, avg_bits = planner.design_strategy()
        
        return self.quantization_plan, avg_bits
    
    def implement_quantization(self, export_to_gguf=False):
        """양자화 개념적 구현"""
        if not self.quantization_plan:
            self.design_quantization_strategy()
        
        if self.model is None:
            self.load_model()
        
        quantizer = Quantizer(
            self.model,
            self.quantization_plan,
            self.weight_stats,
            self.super_weights,
            self.quant_model_dir
        )
        
        quantized_state_dict = quantizer.implement_quantization(export_to_gguf)
        
        return quantized_state_dict
    
    def validate_quantization(self, validation_samples):
        """양자화 모델 검증"""
        if not self.quantization_plan:
            self.design_quantization_strategy()
        
        if self.model is None:
            self.load_model()
        
        quantizer = Quantizer(
            self.model,
            self.quantization_plan,
            self.weight_stats,
            self.super_weights,
            self.quant_model_dir
        )
        
        quantizer.validate_quantization(validation_samples, self.tokenizer)
    
    def run_full_pipeline(self, sample_texts=None, validation_samples=None):
        """전체 양자화 파이프라인 실행"""
        print(f"\n== 동적 양자화 파이프라인 시작 ==")
        print(f"모델: {self.model_name}")
        print(f"출력 디렉토리: {self.output_dir}")
        
        start_time = time.time()
        
        # 1. 모델 로드
        self.load_model()
        
        # 2. 모델 아키텍처 분석
        self.analyze_architecture()
        
        # 3. 레이어별 중요도 분석
        self.analyze_layer_importance(sample_texts)
        
        # 4. 양자화 전략 설계
        self.design_quantization_strategy()
        
        # 5. 양자화 개념적 구현
        self.implement_quantization()
        
        # 6. 양자화 검증 (제공된 경우)
        if validation_samples:
            self.validate_quantization(validation_samples)
        
        elapsed_time = time.time() - start_time
        print(f"\n== 양자화 파이프라인 완료 ==")
        print(f"총 소요 시간: {elapsed_time:.2f}초 ({elapsed_time/60:.2f}분)")
        print(f"결과 파일: {self.output_dir}")
        
        return {
            "model_name": self.model_name,
            "output_dir": self.output_dir,
            "total_params": self.total_params,
            "super_weights": self.super_weights,
            "quantization_plan": self.quantization_plan,
            "elapsed_time": elapsed_time
        }