#!/usr/bin/env python3
# visualization.py - 시각화 관련 기능

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

class VisualizationUtils:
    """시각화 유틸리티 클래스"""
    
    def __init__(self, output_dir="visualizations"):
        """
        시각화 유틸리티 초기화
        
        Args:
            output_dir (str): 출력 디렉토리
        """
        self.output_dir = output_dir
        
        # 한글 폰트 설정 (가능한 경우)
        try:
            plt.rcParams['font.family'] = 'Noto Sans CJK JP'  # 또는 시스템에 설치된 다른 한글 폰트
        except:
            print("경고: 한글 폰트 설정 불가. 기본 폰트 사용")
        
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_layer_distribution(self, layer_type_sizes, total_params, title):
        """레이어 분포 시각화
        
        Args:
            layer_type_sizes (dict): 레이어 타입별 파라미터 수
            total_params (int): 총 파라미터 수
            title (str): 그래프 제목
        """
        # 파이 차트 생성
        plt.figure(figsize=(12, 8))
        labels = []
        sizes = []
        
        for layer_type, size in sorted(layer_type_sizes.items(), key=lambda x: x[1], reverse=True):
            if size / total_params > 0.01:  # 1% 이상인 레이어만 표시
                labels.append(f"{layer_type} ({size/total_params*100:.1f}%)")
                sizes.append(size)
            else:
                # 나머지는 "기타"로 통합
                if "기타" not in labels:
                    labels.append("기타")
                    sizes.append(size)
                else:
                    idx = labels.index("기타")
                    sizes[idx] += size
        
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
        plt.title(title)
        plt.savefig(os.path.join(self.output_dir, "layer_distribution.png"))
        plt.close()
    
    def plot_weight_distributions(self, model, important_layers, super_weights):
        """중요 레이어의 가중치 분포 시각화
        
        Args:
            model: 모델 객체
            important_layers (list): 중요 레이어 목록
            super_weights (list): 슈퍼 웨이트 정보
        """
        # 히스토그램 생성
        plt.figure(figsize=(15, 10))
        for i, layer_name in enumerate(important_layers, 1):
            # 해당 모듈 가져오기
            module = None
            for name, mod in model.named_modules():
                if name == layer_name and hasattr(mod, 'weight'):
                    module = mod
                    break
            
            if module is None:
                continue
                
            # 가중치 가져오기
            weight = module.weight.detach().float().cpu().numpy().flatten()
            
            # 아웃라이어 제외한 범위 계산 (디스플레이 목적)
            lower = np.percentile(weight, 1)
            upper = np.percentile(weight, 99)
            
            # 서브플롯 생성
            plt.subplot(2, 3, i)
            plt.hist(weight, bins=100, range=(lower, upper), alpha=0.7)
            plt.title(f"{layer_name.split('.')[-2]}\n{layer_name}")
            plt.xlabel("Weight Value")
            plt.ylabel("Frequency")
            
            # 슈퍼 웨이트 표시
            for sw in super_weights:
                if sw['layer'] == layer_name:
                    plt.axvline(x=sw['value'], color='r', linestyle='--', 
                                label=f"Super Weight: {sw['value']:.4f}")
                    plt.legend()
                    break
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "weight_distributions.png"))
        plt.close()
    
    def plot_quantization_plan(self, bit_distribution, total_params, title):
        """양자화 계획 시각화
        
        Args:
            bit_distribution (dict): 비트별 파라미터 수
            total_params (int): 총 파라미터 수
            title (str): 그래프 제목
        """
        # 파이 차트 생성
        plt.figure(figsize=(10, 8))
        labels = []
        sizes = []
        
        for bits, params in sorted(bit_distribution.items()):
            percentage = params / total_params * 100
            labels.append(f"{bits}비트 ({percentage:.1f}%)")
            sizes.append(params)
        
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
        plt.title(title)
        plt.savefig(os.path.join(self.output_dir, "quantization_plan.png"))
        plt.close()
    
    def plot_layer_importance(self, layer_importance, layer_sizes):
        """레이어 중요도 시각화
        
        Args:
            layer_importance (dict): 레이어별 중요도 점수
            layer_sizes (dict): 레이어별 크기 정보
        """
        # 산점도 생성 (x축: 레이어 크기, y축: 중요도)
        plt.figure(figsize=(12, 8))
        
        x = []  # 레이어 크기
        y = []  # 중요도 점수
        s = []  # 점 크기
        c = []  # 색상
        annotations = []  # 레이어 이름
        
        # 데이터 준비
        for name, importance in layer_importance.items():
            size = layer_sizes.get(name, 0)
            if size == 0:
                continue
            
            x.append(size)
            y.append(importance)
            s.append(np.log1p(size) * 5)  # 로그 스케일로 점 크기 조정
            
            # 레이어 타입별 색상 할당
            if 'attention' in name.lower():
                c.append('red')
            elif 'ffn' in name.lower() or 'mlp' in name.lower():
                c.append('blue')
            elif 'embed' in name.lower():
                c.append('green')
            elif 'norm' in name.lower():
                c.append('orange')
            else:
                c.append('gray')
            
            # 레이블 간소화
            simple_name = name.split('.')[-1]
            if len(name.split('.')) > 2:
                layer_num = name.split('.')[-2]
                if layer_num.isdigit():
                    simple_name = f"L{layer_num}.{simple_name}"
            
            annotations.append(simple_name)
        
        # 산점도 그리기
        scatter = plt.scatter(x, y, s=s, c=c, alpha=0.7)
        
        # 중요 레이어에 주석 추가
        for i, (xi, yi, annotation) in enumerate(zip(x, y, annotations)):
            if yi > np.percentile(y, 80):  # 상위 20% 중요도만 레이블 표시
                plt.annotate(
                    annotation,
                    (xi, yi),
                    textcoords="offset points",
                    xytext=(0, 5),
                    ha='center'
                )
        
        plt.xscale('log')  # x축 로그 스케일
        plt.xlabel('Layer Size (Parameters)')
        plt.ylabel('Importance Score')
        plt.title('Layer Importance vs. Size')
        plt.grid(True, alpha=0.3)
        
        # 범례 추가
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Attention'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='FFN/MLP'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8, label='Embedding'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=8, label='LayerNorm'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, label='Other')
        ]
        plt.legend(handles=legend_elements, loc='upper left')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "layer_importance.png"))
        plt.close()