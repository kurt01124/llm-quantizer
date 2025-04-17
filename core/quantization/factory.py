#!/usr/bin/env python3
# core/quantization/factory.py - 양자화 팩토리

from typing import Dict, Any, Optional

from .base import QuantizationBase
from .llamacpp import LlamaCppQuantization
from .awq import AWQQuantization
from .gptq import GPTQQuantization

class QuantizationFactory:
    """양자화 방식 팩토리 클래스"""
    
    @staticmethod
    def get_quantizer(method: str, 
                     model: Any, 
                     output_dir: str, 
                     device: str = "cuda") -> QuantizationBase:
        """
        지정된 방식의 양자화 인스턴스 반환
        
        Args:
            method (str): 양자화 방식 ('llamacpp', 'awq', 'gptq')
            model (Any): 양자화할 모델
            output_dir (str): 출력 디렉토리
            device (str): 사용할 디바이스
            
        Returns:
            QuantizationBase: 양자화 인스턴스
        """
        if method.lower() == 'llamacpp':
            return LlamaCppQuantization(model, output_dir, device)
        elif method.lower() == 'awq':
            return AWQQuantization(model, output_dir, device)
        elif method.lower() == 'gptq':
            return GPTQQuantization(model, output_dir, device)
        else:
            raise ValueError(f"지원하지 않는 양자화 방식: {method}")
    
    @staticmethod
    def get_available_methods() -> Dict[str, str]:
        """
        사용 가능한 양자화 방식 목록
        
        Returns:
            Dict[str, str]: 방식 이름 및 설명
        """
        return {
            'llamacpp': 'llama.cpp 기반 양자화 (GGUF 형식)',
            'awq': 'Activation-aware Weight Quantization (AWQ)',
            'gptq': 'GPTQ (Generative Pre-trained Transformer Quantization)'
        }
    
    @staticmethod
    def get_method_description(method: str) -> str:
        """
        양자화 방식 설명 반환
        
        Args:
            method (str): 양자화 방식
            
        Returns:
            str: 방식 설명
        """
        descriptions = {
            'llamacpp': (
                'llama.cpp 기반 양자화 (GGUF 형식).\n'
                '특징: 캘리브레이션 지원, 다양한 비트 수 지원 (2~8비트), '
                '레이어별 다른 양자화 적용 가능.\n'
                '장점: 메모리 사용량 최적화, 높은 호환성.\n'
                '단점: 추론 속도가 상대적으로 느릴 수 있음.'
            ),
            'awq': (
                'Activation-aware Weight Quantization (AWQ).\n'
                '특징: 활성화 인식 양자화, 주로 4비트 양자화.\n'
                '장점: 높은 정확도 유지, 빠른 추론 속도.\n'
                '단점: 더 많은 계산 리소스 필요, 캘리브레이션 필수.'
            ),
            'gptq': (
                'GPTQ (Generative Pre-trained Transformer Quantization).\n'
                '특징: 학습 후 양자화, 주로 3~4비트 양자화.\n'
                '장점: 높은 압축률, 적은 성능 손실.\n'
                '단점: 양자화 과정이 느림, 많은 GPU 메모리 필요.'
            )
        }
        
        return descriptions.get(method.lower(), f"알 수 없는 양자화 방식: {method}")