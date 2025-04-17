#!/usr/bin/env python3
# architecture_analyzer.py - 모델 아키텍처 분석

import os
import json
from collections import defaultdict
from tqdm import tqdm
import re

from utils.visualization import VisualizationUtils

class ArchitectureAnalyzer:
    """모델 아키텍처 분석 클래스"""
    
    def __init__(self, model, output_dir="architecture_analysis"):
        """
        아키텍처 분석기 초기화
        
        Args:
            model: 분석할 모델 객체
            output_dir (str): 결과 저장 디렉토리
        """
        self.model = model
        self.output_dir = output_dir
        self.layer_sizes = defaultdict(int)
        self.total_params = 0
        self.moe_info = {}
        
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
    
    def analyze(self):
        """모델 아키텍처 분석 수행
        
        Returns:
            tuple: (layer_sizes, total_params, moe_info)
        """
        print("\n모델 아키텍처 분석 중...")
        
        # 레이어별 파라미터 수 분석
        layer_types = defaultdict(int)
        self.layer_sizes = defaultdict(int)
        self.total_params = 0
        self.moe_info = {}
        
        # 진행 상황 표시
        modules = list(self.model.named_modules())
        for name, module in tqdm(modules, desc="아키텍처 분석"):
            if hasattr(module, 'weight') and hasattr(module.weight, 'shape'):
                # 안전하게 레이어 타입 추출
                parts = name.split('.')
                layer_type = parts[-2] if len(parts) > 1 else name
                
                params = module.weight.numel()
                layer_types[layer_type] += 1
                self.layer_sizes[name] = params
                self.total_params += params
                
                # MoE 구조 확인
                if 'experts' in name or hasattr(module, 'num_experts'):
                    num_experts = getattr(module, 'num_experts', 'unknown')
                    self.moe_info[name] = {
                        'num_experts': num_experts,
                        'params': params
                    }
        
        # 레이어 타입별 비율 계산
        print("\n레이어 타입별 파라미터 비율:")
        for layer_type, count in sorted(layer_types.items()):
            # 안전하게 합계 계산
            size = sum(self.layer_sizes[name] for name in self.layer_sizes 
                    if len(name.split('.')) > 1 and name.split('.')[-2] == layer_type)
            percentage = size / self.total_params * 100
            print(f"  {layer_type}: {count}개 레이어, {size:,} 파라미터 ({percentage:.2f}%)")
        
        # MoE 구조 요약
        if self.moe_info:
            print("\nMoE 구조 요약:")
            moe_params = sum(info['params'] for info in self.moe_info.values())
            print(f"  MoE 레이어: {len(self.moe_info)}개")
            print(f"  MoE 파라미터: {moe_params:,} ({moe_params/self.total_params*100:.2f}%)")
        
        # 레이어 구조 시각화 및 저장
        self._visualize_layer_distribution()
        
        # 분석 결과 저장
        self._save_analysis_results(layer_types)
        
        return self.layer_sizes, self.total_params, self.moe_info
    
    def _visualize_layer_distribution(self):
        """레이어 분포를 시각화"""
        # 레이어 타입별 파라미터 수 집계
        layer_type_sizes = defaultdict(int)
        for name, size in self.layer_sizes.items():
            layer_type = name.split('.')[-2] if '.' in name else name
            layer_type_sizes[layer_type] += size
        
        # 시각화 유틸리티 사용
        viz = VisualizationUtils(self.output_dir)
        viz.plot_layer_distribution(
            layer_type_sizes, 
            self.total_params,
            f'레이어 타입별 파라미터 분포 (총 {self.total_params:,} 파라미터)'
        )
    
    def _save_analysis_results(self, layer_types):
        """분석 결과 저장"""
        with open(os.path.join(self.output_dir, "architecture_analysis.json"), 'w') as f:
            architecture_info = {
                "total_params": self.total_params,
                "layer_types": {k: v for k, v in layer_types.items()},
                "moe_info": self.moe_info
            }
            json.dump(architecture_info, f, indent=2)
    
    def get_layer_count_by_type(self):
        """레이어 타입별 개수 반환"""
        layer_type_counts = defaultdict(int)
        for name in self.layer_sizes.keys():
            parts = name.split('.')
            layer_type = parts[-2] if len(parts) > 1 else name
            layer_type_counts[layer_type] += 1
        
        return dict(layer_type_counts)
    
    def get_model_structure_summary(self):
        """모델 구조 요약 정보 반환"""
        # 레이어 타입별 분류
        layer_types = defaultdict(list)
        for name in self.layer_sizes.keys():
            parts = name.split('.')
            layer_type = parts[-2] if len(parts) > 1 else name
            layer_types[layer_type].append(name)
        
        # 트랜스포머 블록 수 추정
        transformer_blocks = 0
        for name in self.layer_sizes.keys():
            match = re.search(r'layers?\.(\d+)', name)
            if match:
                block_num = int(match.group(1)) + 1
                transformer_blocks = max(transformer_blocks, block_num)
        
        return {
            "total_params": self.total_params,
            "estimated_transformer_blocks": transformer_blocks,
            "has_moe": bool(self.moe_info),
            "layer_type_counts": self.get_layer_count_by_type()
        }