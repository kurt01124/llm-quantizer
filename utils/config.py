#!/usr/bin/env python3
# config.py - 설정 관련 기능

import json
import os
from typing import Dict, Any

def parse_config(config_path: str) -> Dict[str, Any]:
    """설정 파일 파싱
    
    Args:
        config_path (str): 설정 파일 경로
        
    Returns:
        dict: 설정 정보
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    return config

def save_config(config: Dict[str, Any], output_path: str) -> None:
    """설정 파일 저장
    
    Args:
        config (dict): 설정 정보
        output_path (str): 출력 파일 경로
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"설정이 {output_path}에 저장되었습니다")

def generate_default_config(model_name: str, output_dir: str = "quantized_model") -> Dict[str, Any]:
    """기본 설정 생성
    
    Args:
        model_name (str): 모델 이름
        output_dir (str): 출력 디렉토리
        
    Returns:
        dict: 기본 설정 정보
    """
    return {
        "model": model_name,
        "output_dir": output_dir,
        "device": "cuda",
        "generate_calibration": True,
        "cal_samples": 20,
        "skip_architecture": False,
        "skip_importance": False,
        "skip_conceptual": False,
        "quantization": {
            "default_bits": 4,
            "embedding_bits": 8,
            "lm_head_bits": 8,
            "norm_bits": 16,
            "attention_bits": 4,
            "super_weight_bits": 6
        },
        "inference": {
            "temp": 0.8,
            "repeat_penalty": 1.2,
            "repeat_last_n": 128,
            "presence_penalty": 0.2,
            "frequency_penalty": 0.2
        }
    }