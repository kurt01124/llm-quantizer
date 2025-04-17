#!/usr/bin/env python3
# llama_cpp_python.py - llama-cpp-python 래퍼

import time
from typing import List, Dict, Any, Optional, Union, Tuple

class LlamaCppInference:
    """llama-cpp-python을 사용한 모델 추론 클래스"""
    
    def __init__(
        self,
        model_path: str,
        n_ctx: int = 4096,
        n_batch: int = 512,
        n_gpu_layers: int = -1,
        seed: int = -1,
        verbose: bool = False
    ):
        """
        추론 클래스 초기화
        
        Args:
            model_path (str): GGUF 모델 파일 경로
            n_ctx (int): 컨텍스트 크기
            n_batch (int): 배치 크기
            n_gpu_layers (int): GPU에 로드할 레이어 수 (-1: 모두)
            seed (int): 랜덤 시드 (-1: 랜덤)
            verbose (bool): 상세 로깅 여부
        """
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_batch = n_batch
        self.n_gpu_layers = n_gpu_layers
        self.seed = seed
        self.verbose = verbose
        self.model = None
        
        self._load_model()
    
    def _load_model(self) -> None:
        """모델 로드"""
        try:
            from llama_cpp import Llama
        except ImportError:
            print("llama-cpp-python 라이브러리가 설치되지 않았습니다.")
            print("설치하려면: pip install llama-cpp-python")
            raise
        
        print(f"모델 로드 중: {self.model_path}")
        start_time = time.time()
        
        # llama-cpp-python 모델 로드
        self.model = Llama(
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            n_batch=self.n_batch,
            n_gpu_layers=self.n_gpu_layers,
            seed=self.seed,
            verbose=self.verbose
        )
        
        load_time = time.time() - start_time
        print(f"모델 로드 완료: {load_time:.2f}초")
        print(f"모델 정보: n_ctx={self.model.n_ctx()}, vocab_size={self.model.n_vocab()}")
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
        repeat_last_n: int = 128,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        mirostat_mode: int = 0,
        mirostat_tau: float = 5.0,
        mirostat_eta: float = 0.1,
        stop: Optional[List[str]] = None,
        stream: bool = False
    ) -> Union[str, List[Dict[str, Any]]]:
        """텍스트 생성
        
        Args:
            prompt (str): 입력 프롬프트
            max_tokens (int): 생성할 최대 토큰 수
            temperature (float): 온도 (높을수록 무작위적)
            top_p (float): Nucleus sampling 임계값
            top_k (int): Top-K sampling 임계값
            repeat_penalty (float): 반복 페널티
            repeat_last_n (int): 반복 검사할 이전 토큰 수
            presence_penalty (float): 존재 기반 페널티
            frequency_penalty (float): 빈도 기반 페널티
            mirostat_mode (int): Mirostat 알고리즘 모드 (0=비활성화, 1=v1, 2=v2)
            mirostat_tau (float): Mirostat 목표 엔트로피
            mirostat_eta (float): Mirostat 학습률
            stop (list): 생성 중단 토큰 목록
            stream (bool): 스트리밍 모드 사용 여부
            
        Returns:
            str 또는 list: 생성된 텍스트 또는 스트리밍 결과 목록
        """
        if self.model is None:
            self._load_model()
        
        # 생성 옵션 설정
        kwargs = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repeat_penalty": repeat_penalty,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "last_n_tokens_size": repeat_last_n,
            "stop": stop or [],
            "stream": stream
        }
        
        # Mirostat이 활성화된 경우
        if mirostat_mode > 0:
            kwargs.update({
                "mirostat_mode": mirostat_mode,
                "mirostat_tau": mirostat_tau,
                "mirostat_eta": mirostat_eta
            })
        
        # 생성 실행
        start_time = time.time()
        
        if stream:
            # 스트리밍 모드
            stream_results = []
            for chunk in self.model(**kwargs):
                stream_results.append(chunk)
            gen_time = time.time() - start_time
            
            # 생성 시간 출력
            tokens_generated = sum(len(chunk.get("tokens", [])) for chunk in stream_results)
            if tokens_generated > 0:
                tokens_per_second = tokens_generated / gen_time
                print(f"생성 완료: {tokens_generated}토큰, {gen_time:.2f}초 ({tokens_per_second:.2f} 토큰/초)")
            
            return stream_results
        else:
            # 일반 모드
            result = self.model(**kwargs)
            gen_time = time.time() - start_time
            
            # 생성 시간 출력
            tokens = result.get("tokens", [])
            if tokens and len(tokens) > 0:
                tokens_per_second = len(tokens) / gen_time
                print(f"생성 완료: {len(tokens)}토큰, {gen_time:.2f}초 ({tokens_per_second:.2f} 토큰/초)")
            
            return result.get("choices", [{}])[0].get("text", "")
    
    def anti_repetition_settings(self) -> Dict[str, Any]:
        """반복 출력 방지를 위한 추천 설정
        
        Returns:
            dict: 추천 설정 정보
        """
        return {
            "temperature": 0.8,
            "top_p": 0.9,
            "top_k": 40,
            "repeat_penalty": 1.2,
            "repeat_last_n": 128,
            "presence_penalty": 0.2,
            "frequency_penalty": 0.2,
            "mirostat_mode": 2,
            "mirostat_tau": 5.0,
            "mirostat_eta": 0.1
        }
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        **generation_kwargs
    ) -> Dict[str, Any]:
        """챗 완성 API
        
        Args:
            messages (list): 메시지 목록 (OpenAI 형식)
            max_tokens (int): 생성할 최대 토큰 수
            **generation_kwargs: 추가 생성 옵션
            
        Returns:
            dict: 챗 완성 결과
        """
        if self.model is None:
            self._load_model()
        
        # 메시지를 프롬프트로 변환
        try:
            formatted = self.model.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                **generation_kwargs
            )
            return formatted
        except AttributeError:
            # 구 버전 라이브러리 호환성 처리
            print("경고: 현재 llama-cpp-python 버전이 create_chat_completion을 지원하지 않습니다.")
            print("수동으로 포맷팅하여 실행합니다.")
            
            # 간단한 포맷팅
            prompt = ""
            for message in messages:
                role = message.get("role", "user")
                content = message.get("content", "")
                
                if role == "system":
                    prompt += f"<s>[SYSTEM] {content}</s>\n"
                elif role == "user":
                    prompt += f"<s>[USER] {content}</s>\n"
                elif role == "assistant":
                    prompt += f"<s>[ASSISTANT] {content}</s>\n"
            
            prompt += "<s>[ASSISTANT] "
            
            # 일반 생성 사용
            result = self.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                **generation_kwargs
            )
            
            # OpenAI 형식 응답 구성
            return {
                "choices": [{
                    "message": {
                        "role": "assistant", 
                        "content": result
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": len(prompt),
                    "completion_tokens": len(result),
                    "total_tokens": len(prompt) + len(result)
                }
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환
        
        Returns:
            dict: 모델 정보
        """
        if self.model is None:
            self._load_model()
        
        return {
            "model_path": self.model_path,
            "n_ctx": self.model.n_ctx(),
            "n_batch": self.n_batch,
            "n_gpu_layers": self.n_gpu_layers,
            "vocab_size": self.model.n_vocab(),
            "seed": self.seed
        }