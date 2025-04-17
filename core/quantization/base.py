#!/usr/bin/env python3
# core/quantization/base.py - 양자화 기본 클래스

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional
import torch

class QuantizationBase(ABC):
    """양자화 추상 기본 클래스"""
    
    def __init__(self, model, output_dir: str, device: str = "cuda"):
        """
        양자화 기본 클래스 초기화
        
        Args:
            model: 양자화할 모델
            output_dir (str): 출력 디렉토리
            device (str): 사용할 디바이스
        """
        self.model = model
        self.output_dir = output_dir
        self.device = device
        
    @abstractmethod
    def quantize(self, quantization_config: Dict[str, Any]) -> Tuple[Any, str]:
        """
        모델 양자화 수행
        
        Args:
            quantization_config (Dict[str, Any]): 양자화 설정
            
        Returns:
            Tuple[Any, str]: (양자화된 모델, 모델 경로)
        """
        pass
    
    @abstractmethod
    def generate_quantization_config(self, 
                                    layer_importance: Dict[str, float],
                                    super_weights: List[Dict[str, Any]],
                                    layer_sizes: Dict[str, int],
                                    **kwargs) -> Dict[str, Any]:
        """
        양자화 설정 생성
        
        Args:
            layer_importance (Dict[str, float]): 레이어별 중요도
            super_weights (List[Dict[str, Any]]): 슈퍼 웨이트 정보
            layer_sizes (Dict[str, int]): 레이어별 크기
            **kwargs: 추가 인자
            
        Returns:
            Dict[str, Any]: 양자화 설정
        """
        pass
    
    @abstractmethod
    def supports_calibration(self) -> bool:
        """
        캘리브레이션 지원 여부
        
        Returns:
            bool: 캘리브레이션 지원 여부
        """
        pass
    
    @abstractmethod
    def get_method_name(self) -> str:
        """
        양자화 방식 이름
        
        Returns:
            str: 양자화 방식 이름
        """
        pass
    
    @abstractmethod
    def get_optimal_bit_allocation(self, 
                                  layer_name: str, 
                                  importance: float, 
                                  is_super_weight: bool) -> int:
        """
        최적 비트 할당
        
        Args:
            layer_name (str): 레이어 이름
            importance (float): 중요도 점수
            is_super_weight (bool): 슈퍼 웨이트 여부
            
        Returns:
            int: 최적 비트 수
        """
        pass
    
    def prepare_model_for_quantization(self) -> Any:
        """
        양자화를 위한 모델 준비 (기본 구현)
        
        Returns:
            Any: 준비된 모델
        """
        # 기본적으로는 모델을 CPU로 이동하고 평가 모드로 설정
        return self.model.cpu().eval()
    
    def get_default_quantization_config(self) -> Dict[str, Any]:
        """
        기본 양자화 설정 반환 (기본 구현)
        
        Returns:
            Dict[str, Any]: 기본 양자화 설정
        """
        return {
            "default_bits": 4,
            "embedding_bits": 8,
            "lm_head_bits": 8,
            "norm_bits": 16,
            "attention_bits": 4,
            "super_weight_bits": 6
        }
    
    def estimate_size_reduction(self, 
                               quantization_config: Dict[str, Any], 
                               total_params: int,
                               layer_sizes: Dict[str, int]) -> float:
        """
        양자화 후 크기 감소 예상 (기본 구현)
        
        Args:
            quantization_config (Dict[str, Any]): 양자화 설정
            total_params (int): 총 파라미터 수
            layer_sizes (Dict[str, int]): 레이어별 크기
            
        Returns:
            float: 예상 크기 비율 (원본 대비)
        """
        # 기본 FP16 대비 예상 크기 비율
        avg_bits = quantization_config.get("default_bits", 4)
        return avg_bits / 16.0  # FP16 기준