#!/usr/bin/env python3
# model_loader.py - 모델 로딩 관련 기능

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class ModelLoader:
    """모델 로딩 클래스"""
    
    def __init__(self, model_name, device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        모델 로더 초기화
        
        Args:
            model_name (str): 로드할 모델 이름 (Hugging Face ID)
            device (str): 사용할 디바이스 (기본값: cuda 또는 cpu)
        """
        self.model_name = model_name
        self.device = device
        self.tokenizer = None
        self.model = None
    
    def load_model(self):
        """모델 및 토크나이저 로드
        
        Returns:
            tuple: (모델, 토크나이저)
        """
        print(f"\n모델 로드 중: {self.model_name}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map=self.device
            )
            
            print(f"모델 로드 완료: {type(self.model).__name__}")
            return self.model, self.tokenizer
        
        except Exception as e:
            print(f"모델 로드 중 오류 발생: {str(e)}")
            raise
    
    def get_model_info(self):
        """모델 정보 반환
        
        Returns:
            dict: 모델 정보를 담은 딕셔너리
        """
        if self.model is None:
            self.load_model()
        
        num_params = sum(p.numel() for p in self.model.parameters())
        
        return {
            "model_name": self.model_name,
            "model_type": type(self.model).__name__,
            "num_params": num_params,
            "device": self.device,
            "dtype": str(next(self.model.parameters()).dtype)
        }