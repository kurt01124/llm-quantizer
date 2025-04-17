#!/usr/bin/env python3
# quantization_planner.py - 양자화 전략 설계

import os
import json
import re
from collections import defaultdict
from tqdm import tqdm

from utils.visualization import VisualizationUtils
from utils.script_generator import ScriptGenerator

class QuantizationPlanner:
    """양자화 전략 설계 클래스"""
    
    def __init__(self, model_name, layer_sizes, total_params, layer_importance, super_weights, 
                 output_dir="quantization_plan", moe_info=None):
        """
        양자화 플래너 초기화
        
        Args:
            model_name (str): 모델 이름
            layer_sizes (dict): 레이어별 크기 정보
            total_params (int): 총 파라미터 수
            layer_importance (dict): 레이어별 중요도 점수
            super_weights (list): 슈퍼 웨이트 정보
            output_dir (str): 결과 저장 디렉토리
            moe_info (dict): MoE 구조 정보 (선택 사항)
        """
        self.model_name = model_name
        self.layer_sizes = layer_sizes
        self.total_params = total_params
        self.layer_importance = layer_importance
        self.super_weights = super_weights
        self.output_dir = output_dir
        self.moe_info = moe_info or {}
        
        self.quantization_plan = {}
        
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
    
    def design_strategy(self):
        """양자화 전략 설계
        
        Returns:
            tuple: (quantization_plan, avg_bits)
        """
        print("\n양자화 전략 설계 중...")
        
        # 레이어별 양자화 비트 할당
        self.quantization_plan = {}
        total_bits = 0
        
        # 정규화된 중요도 점수 계산
        if self.layer_importance:
            max_importance = max(self.layer_importance.values())
            normalized_importance = {name: score/max_importance for name, score in self.layer_importance.items()}
        else:
            normalized_importance = {}
        
        # 레이어별 특성 분석 및 양자화 비트 할당
        for name, params in tqdm(sorted(self.layer_sizes.items(), key=lambda x: x[1], reverse=True), 
                                desc="양자화 계획 설계"):
            # 패턴 매칭으로 레이어 타입 식별
            is_embedding = 'embed' in name.lower() or 'wte' in name.lower()
            is_lm_head = 'lm_head' in name.lower() or 'output' in name.lower() or 'classifier' in name.lower()
            is_attention = any(x in name.lower() for x in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'attention'])
            is_down_proj = 'down_proj' in name.lower() or 'down_proj' in name.lower()
            is_norm = 'norm' in name.lower() or 'ln' in name.lower()
            is_router = 'router' in name.lower()
            is_expert = 'expert' in name.lower() and bool(self.moe_info)
            
            # 레이어 번호 추출 시도
            layer_num = -1
            match = re.search(r'layers?\.(\d+)', name)
            if match:
                layer_num = int(match.group(1))
            
            # 중요도 점수 가져오기
            importance = normalized_importance.get(name, 0.0)
            
            # 슈퍼 웨이트 여부 확인
            is_super_weight_layer = False
            for sw in self.super_weights[:10]:  # 상위 10개만 고려
                if sw['layer'] == name:
                    is_super_weight_layer = True
                    break
            
            # 양자화 비트 결정
            if is_embedding or is_lm_head:
                bits = 8  # 임베딩 및 출력 레이어
            elif is_norm or is_router:
                bits = 16  # 레이어 정규화 및 라우터
            elif is_super_weight_layer:
                bits = 6  # 슈퍼 웨이트를 포함한 레이어
            elif is_down_proj and (layer_num >= 0 and layer_num <= 5):
                bits = 6 if importance > 0.7 else 4  # 초기 down_proj 레이어
            elif is_attention and (layer_num >= 0 and layer_num <= 3):
                bits = 6  # 초기 어텐션 레이어
            elif is_attention:
                bits = 4  # 나머지 어텐션 레이어
            elif is_expert:
                bits = 1.58  # MoE 전문가 가중치
            else:
                bits = 4 if importance > 0.5 else 2  # 기타 레이어
            
            self.quantization_plan[name] = bits
            total_bits += bits * params
        
        # 평균 비트 수 계산
        avg_bits = total_bits / self.total_params
        
        print(f"예상 평균 비트 수: {avg_bits:.2f} 비트/가중치")
        print(f"예상 압축 후 크기: {total_bits/8/1024/1024/1024:.2f} GB")
        
        # 양자화 계획 요약
        bit_distribution = self._summarize_quantization_plan()
        
        # 양자화 계획 시각화
        self._visualize_quantization_plan(bit_distribution)
        
        # 양자화 계획 저장
        self._save_quantization_plan(bit_distribution, avg_bits, total_bits)
        
        # llama.cpp 양자화 명령어 생성
        self._generate_quantization_command()
        
        return self.quantization_plan, avg_bits
    
    def _summarize_quantization_plan(self):
        """양자화 계획 요약"""
        bit_distribution = defaultdict(int)
        for name, bits in self.quantization_plan.items():
            params = self.layer_sizes.get(name, 0)
            bit_distribution[bits] += params
        
        print("\n양자화 계획 요약:")
        for bits, params in sorted(bit_distribution.items()):
            percentage = params / self.total_params * 100
            print(f"  {bits}비트: {params:,} 파라미터 ({percentage:.2f}%)")
        
        return bit_distribution
    
    def _visualize_quantization_plan(self, bit_distribution):
        """양자화 계획 시각화"""
        viz = VisualizationUtils(self.output_dir)
        viz.plot_quantization_plan(
            bit_distribution, 
            self.total_params,
            f'양자화 계획 (평균: {sum([b*s for b,s in bit_distribution.items()])/self.total_params:.2f} 비트/가중치)'
        )
    
    def _save_quantization_plan(self, bit_distribution, avg_bits, total_bits):
        """양자화 계획 저장"""
        with open(os.path.join(self.output_dir, "quantization_plan.json"), 'w') as f:
            plan_info = {
                "avg_bits": avg_bits,
                "total_size_gb": total_bits/8/1024/1024/1024,
                "bit_distribution": {str(k): int(v) for k, v in bit_distribution.items()},
                "quantization_plan": {k: float(v) for k, v in self.quantization_plan.items()}
            }
            json.dump(plan_info, f, indent=2)
    
    def _generate_quantization_command(self):
        """llama.cpp 양자화 명령어 생성"""
        # 스크립트 생성기 사용
        script_gen = ScriptGenerator(self.model_name, self.output_dir)
        
        # 슈퍼 웨이트 레이어 목록 추출
        sw_layers = []
        for sw in self.super_weights[:10]:  # 상위 10개만 고려
            layer_pattern = sw['layer']
            if layer_pattern not in sw_layers:
                sw_layers.append(layer_pattern)
        
        # 양자화 명령어 스크립트 생성
        script_gen.generate_quantization_script(sw_layers)
    
    def get_bit_distribution_summary(self):
        """비트 분포 요약 정보 반환"""
        if not self.quantization_plan:
            return {}
            
        bit_distribution = defaultdict(int)
        for name, bits in self.quantization_plan.items():
            params = self.layer_sizes.get(name, 0)
            bit_distribution[bits] += params

        total_bits = sum([bits * params for bits, params in bit_distribution.items()])
        avg_bits = total_bits / self.total_params if self.total_params > 0 else 0
            
        return {
            "distribution": {str(k): int(v) for k, v in bit_distribution.items()},
            "avg_bits": avg_bits,
            "total_size_gb": total_bits/8/1024/1024/1024
        }