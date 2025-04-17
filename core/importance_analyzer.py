#!/usr/bin/env python3
# importance_analyzer.py - 레이어 중요도 분석

import os
import json
import torch
import numpy as np
from tqdm import tqdm

from utils.visualization import VisualizationUtils

class ImportanceAnalyzer:
    """레이어 중요도 분석 클래스"""
    
    def __init__(self, model, tokenizer, layer_sizes, output_dir="importance_analysis", device=None):
        """
        중요도 분석기 초기화
        
        Args:
            model: 분석할 모델 객체
            tokenizer: 토크나이저
            layer_sizes (dict): 레이어별 크기 정보
            output_dir (str): 결과 저장 디렉토리
            device (str): 사용할 디바이스
        """
        self.model = model
        self.tokenizer = tokenizer
        self.layer_sizes = layer_sizes
        self.output_dir = output_dir
        self.device = device or next(model.parameters()).device
        
        self.weight_stats = {}
        self.layer_importance = {}
        self.super_weights = []
        
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
    
    def analyze(self, sample_texts=None, max_samples=3):
        """레이어 중요도 분석 수행
        
        Args:
            sample_texts (list): 활성화 분석용 샘플 텍스트 목록
            max_samples (int): 최대 샘플 수
            
        Returns:
            tuple: (weight_stats, layer_importance, super_weights)
        """
        print("\n레이어별 중요도 분석 중...")
        
        # 가중치 분포 분석
        self._analyze_weight_distributions()
        
        # 활성화 값 분석 (샘플 입력이 제공된 경우)
        if sample_texts and self.tokenizer:
            activation_stats = self._analyze_activations(sample_texts, max_samples)
            self._adjust_importance_by_activations(activation_stats)
        
        # 슈퍼 웨이트 정보 시각화
        self._visualize_weight_distributions()
        
        # 중요도 분석 결과 저장
        self._save_analysis_results()
        
        return self.weight_stats, self.layer_importance, self.super_weights
    
    def _analyze_weight_distributions(self):
        """가중치 분포 분석"""
        # 진행 상황 표시
        modules = [(name, module) for name, module in self.model.named_modules() 
                  if hasattr(module, 'weight') and hasattr(module.weight, 'shape')]
        
        for name, module in tqdm(modules, desc="가중치 분석"):
            weight = module.weight.detach().float().cpu().numpy()
            
            # 통계 계산
            stats = {
                'mean': float(np.mean(weight)),
                'std': float(np.std(weight)),
                'min': float(np.min(weight)),
                'max': float(np.max(weight)),
                'abs_max': float(np.max(np.abs(weight))),
                'sparsity': float(np.mean(weight == 0)),
                'shape': [int(x) for x in weight.shape],
            }
            
            self.weight_stats[name] = stats
            
            # 슈퍼 웨이트 후보 식별 (큰 가중치 찾기)
            is_candidate = 'down_proj' in name or 'o_proj' in name
            if is_candidate:
                # 상위 0.01% 가중치 찾기
                flat_weights = weight.flatten()
                threshold = np.percentile(np.abs(flat_weights), 99.99)
                outliers = np.where(np.abs(flat_weights) > threshold)[0]
                
                # 가장 큰 가중치의 위치 찾기
                max_idx = np.argmax(np.abs(flat_weights))
                max_val = flat_weights[max_idx]
                max_pos = np.unravel_index(max_idx, weight.shape)
                
                # 슈퍼 웨이트 후보 등록
                self.super_weights.append({
                    'layer': name,
                    'value': float(max_val),
                    'position': [int(x) for x in max_pos],
                    'threshold': float(threshold),
                    'num_outliers': int(len(outliers)),
                })
                
                # 중요도 점수 할당
                importance_score = float(np.abs(max_val) / np.mean(np.abs(flat_weights)))
                self.layer_importance[name] = importance_score
        
        # 슈퍼 웨이트 후보 정렬
        self.super_weights.sort(key=lambda x: abs(x['value']), reverse=True)
        
        # 상위 슈퍼 웨이트 출력
        print("\n슈퍼 웨이트 후보 (상위 5개):")
        for i, sw in enumerate(self.super_weights[:5], 1):
            print(f"{i}. 레이어: {sw['layer']}")
            print(f"   값: {sw['value']:.6f}")
            print(f"   위치: {sw['position']}")
            print(f"   중요도 점수: {self.layer_importance[sw['layer']]:.2f}")
    
    def _analyze_activations(self, sample_texts, max_samples):
        """활성화 값 분석
        
        Args:
            sample_texts (list): 분석할 샘플 텍스트 목록
            max_samples (int): 최대 샘플 수
            
        Returns:
            dict: 활성화 통계 정보
        """
        print("\n활성화 값 분석 중...")
        
        # 샘플 제한
        sample_texts = sample_texts[:max_samples]
        
        # 활성화 값 수집을 위한 딕셔너리
        activation_stats = {}
        
        # 활성화 값 수집을 위한 훅 설정
        hooks = []
        
        def hook_fn(name):
            def fn(module, input, output):
                if isinstance(output, tuple):
                    output = output[0]
                
                # 처음 등록하는 경우 초기화
                if name not in activation_stats:
                    activation_stats[name] = {
                        'abs_max': 0.0,
                        'counts': 0
                    }
                
                # 최대값 업데이트
                curr_max = output.float().abs().max().item()
                activation_stats[name]['abs_max'] = max(
                    activation_stats[name]['abs_max'], 
                    curr_max
                )
                activation_stats[name]['counts'] += 1
                
            return fn
        
        # 훅 등록
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                hooks.append(module.register_forward_hook(hook_fn(name)))
        
        # 샘플별로 추론 실행
        for text in tqdm(sample_texts, desc="활성화 분석"):
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            with torch.no_grad():
                _ = self.model(**inputs)
        
        # 훅 제거
        for hook in hooks:
            hook.remove()
        
        # 활성화 값이 큰 레이어 출력
        print("\n활성화 값이 큰 레이어 (상위 5개):")
        for name, stats in sorted(activation_stats.items(), key=lambda x: x[1]['abs_max'], reverse=True)[:5]:
            print(f"레이어: {name}")
            print(f"  최대 활성화 값: {stats['abs_max']:.6f}")
        
        return activation_stats
    
    def _adjust_importance_by_activations(self, activation_stats):
        """활성화 값 기반 중요도 조정
        
        Args:
            activation_stats (dict): 활성화 통계 정보
        """
        for name, stats in activation_stats.items():
            if name in self.layer_importance:
                self.layer_importance[name] *= (1 + np.log1p(stats['abs_max']))
                print(f"  조정된 중요도 점수: {self.layer_importance[name]:.2f}")
    
    def _visualize_weight_distributions(self):
        """중요 레이어의 가중치 분포 시각화"""
        # 중요 레이어 선택 (슈퍼 웨이트 상위 3개, 및 기타 중요 레이어)
        important_layers = []
        for sw in self.super_weights[:3]:
            important_layers.append(sw['layer'])
        
        # down_proj 레이어 중 일부 추가
        for name in self.weight_stats:
            if 'down_proj' in name and name not in important_layers:
                important_layers.append(name)
                if len(important_layers) >= 6:  # 최대 6개 레이어
                    break
        
        # 시각화 유틸리티 사용
        viz = VisualizationUtils(self.output_dir)
        viz.plot_weight_distributions(
            self.model, 
            important_layers, 
            self.super_weights
        )
    
    def _save_analysis_results(self):
        """분석 결과 저장"""
        with open(os.path.join(self.output_dir, "importance_analysis.json"), 'w') as f:
            importance_info = {
                "super_weights": self.super_weights,
                "layer_importance": {k: float(v) for k, v in self.layer_importance.items()},
            }
            json.dump(importance_info, f, indent=2)
    
    def get_normalized_importance(self):
        """정규화된 중요도 점수 반환"""
        if not self.layer_importance:
            return {}
        
        max_importance = max(self.layer_importance.values())
        return {name: score/max_importance for name, score in self.layer_importance.items()}
    
    def get_super_weight_layers(self, top_n=10):
        """상위 슈퍼 웨이트 레이어 반환"""
        sw_layers = []
        for sw in self.super_weights[:top_n]:
            layer_name = sw['layer']
            if layer_name not in sw_layers:
                sw_layers.append(layer_name)
        return sw_layers