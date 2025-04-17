#!/usr/bin/env python3
# core/quantization/gptq.py - GPTQ 양자화 구현

import os
import json
import re
from typing import Dict, Any, List, Tuple, Optional
import torch
from collections import defaultdict

from .base import QuantizationBase

class GPTQQuantization(QuantizationBase):
    """GPTQ (Generative Pre-trained Transformer Quantization) 구현 클래스"""
    
    def __init__(self, model, output_dir: str, device: str = "cuda"):
        """
        GPTQ 양자화 클래스 초기화
        
        Args:
            model: 양자화할 모델
            output_dir (str): 출력 디렉토리
            device (str): 사용할 디바이스
        """
        super().__init__(model, output_dir, device)
        
        # GPTQ 관련 디렉토리 및 파일 경로
        self.gptq_dir = os.path.join(output_dir, "gptq")
        self.gptq_model_path = os.path.join(output_dir, "model_gptq_quantized")
        
        # 스크립트 저장 경로
        self.script_dir = os.path.join(output_dir, "scripts")
        os.makedirs(self.script_dir, exist_ok=True)
    
    def get_method_name(self) -> str:
        """
        양자화 방식 이름
        
        Returns:
            str: 양자화 방식 이름
        """
        return "gptq"
    
    def supports_calibration(self) -> bool:
        """
        캘리브레이션 지원 여부
        
        Returns:
            bool: 캘리브레이션 지원 여부
        """
        return True
    
    def get_optimal_bit_allocation(self, 
                                  layer_name: str, 
                                  importance: float, 
                                  is_super_weight: bool) -> int:
        """
        최적 비트 할당 (GPTQ는 주로 3-4비트만 지원)
        
        Args:
            layer_name (str): 레이어 이름
            importance (float): 중요도 점수
            is_super_weight (bool): 슈퍼 웨이트 여부
            
        Returns:
            int: 최적 비트 수
        """
        # GPTQ는 보통 4비트 또는 3비트 양자화를 사용
        return 4 if is_super_weight or importance > 0.6 else 3
    
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
        # 슈퍼 웨이트 레이어 추출 (상위 10개)
        super_weight_layers = []
        for sw in super_weights[:10]:
            super_weight_layers.append(sw['layer'])
        
        # 중요 레이어 추출 (중요도가 높은 상위 레이어)
        important_layers = []
        if layer_importance:
            sorted_importance = sorted(layer_importance.items(), key=lambda x: x[1], reverse=True)
            important_layers = [name for name, _ in sorted_importance[:10]]
        
        # GPTQ 특유의 설정
        config = {
            "method": "gptq",
            "bits": 4,  # 기본 비트 수
            "desc_act": True,  # 활성화 함수 설명자 사용
            "group_size": 128,  # 그룹 크기
            "damp_percent": 0.01,  # 댐핑 퍼센트
            "sym": True,  # 대칭 양자화
            "true_sequential": True,  # 순차적 처리
            "block_name_to_quantize": None,  # 양자화할 블록 이름 (None: 전체)
            "super_weight_layers": super_weight_layers,
            "important_layers": important_layers,
            "percdamp": 0.01,  # 퍼센트 댐핑
            "act_order": True,  # 활성화 순서
            "static_groups": False,  # 정적 그룹
            "calibration_needed": True
        }
        
        # 특수 레이어 처리 설정
        special_layers = {
            "lm_head": False,  # LM 헤드는 양자화하지 않음
            "embed_tokens": False,  # 임베딩 토큰은 양자화하지 않음
            "output_projection": False,  # 출력 프로젝션은 양자화하지 않음
        }
        
        # 중요 레이어에 대한 특별 설정
        layer_settings = {}
        for layer in important_layers:
            # 중요도가 높은 레이어는 더 정밀한 양자화 설정
            layer_settings[layer] = {
                "bits": 4,  # 더 높은 비트 수
                "group_size": 64  # 더 작은 그룹 크기
            }
        
        config["layer_settings"] = layer_settings
        config["special_layers"] = special_layers
        
        return config
    
    def quantize(self, quantization_config: Dict[str, Any]) -> Tuple[Any, str]:
        """
        모델 양자화 수행 (스크립트 생성 및 실행)
        
        Args:
            quantization_config (Dict[str, Any]): 양자화 설정
            
        Returns:
            Tuple[Any, str]: (양자화된 모델, 모델 경로)
        """
        # 양자화 스크립트 생성
        script_path = self._generate_quantization_script(quantization_config)
        
        # 개념적 양자화 구현 (실제 스크립트는 별도 실행)
        model_path = self.gptq_model_path
        
        print(f"\n양자화 명령어 스크립트가 생성되었습니다: {script_path}")
        print(f"실제 양자화를 수행하려면 다음 명령어를 실행하세요:")
        print(f"  chmod +x {script_path}")
        print(f"  {script_path}")
        
        # 양자화된 모델은 스크립트 실행 후 생성되므로 None 반환
        return None, model_path
    
    def _generate_quantization_script(self, quantization_config: Dict[str, Any]) -> str:
        """
        GPTQ 양자화 명령어 스크립트 생성
        
        Args:
            quantization_config (Dict[str, Any]): 양자화 설정
            
        Returns:
            str: 생성된 스크립트 경로
        """
        output_path = os.path.join(self.script_dir, "gptq_quantization.sh")
        
        with open(output_path, 'w') as f:
            f.write("#!/bin/bash\n\n")
            f.write("# 이 스크립트는 모델을 GPTQ 양자화하기 위한 명령어입니다.\n\n")

            f.write("set -e\n\n")

            # 모델명 변수 추가
            model_name = getattr(self.model, 'name_or_path', 'model')
            f.write(f"MODEL_NAME=\"{model_name}\"\n\n")
            
            f.write("# 1. Auto-GPTQ 패키지 설치 (필요한 경우)\n")
            f.write("pip install -U auto-gptq packaging optimum\n\n")
            
            # 모델 설정 저장
            f.write("# 2. 양자화 설정 저장\n")
            f.write("cat > gptq_config.json << 'EOL'\n")
            f.write(json.dumps(quantization_config, indent=2))
            f.write("\nEOL\n\n")
            
            # 캘리브레이션 데이터 생성
            f.write("""# 3. 캘리브레이션 데이터 생성
echo "GPTQ 캘리브레이션 데이터 생성 중..."
if [ ! -f "gptq_calibration_data.jsonl" ]; then
  python -c '
import json
import random
import os

def generate_gptq_calibration_data(num_samples=128, tokens_per_sample=512):
    # 다양한 프롬프트 템플릿
    prompts = [
        "인공지능의 미래에 대해 설명해주세요.",
        "파이썬으로 간단한 웹 크롤러를 만들어주세요.",
        "우주 탐험가의 일기를 작성해주세요.",
        "양자 컴퓨팅의 기본 원리를 설명해주세요.",
        "사용자: 안녕하세요, 오늘 기분이 어떠세요?\\n어시스턴트:",
        "기후 변화의 주요 원인과 해결책은 무엇인가요?",
        "자바스크립트로 투두리스트 앱을 만드는 코드를 작성해주세요.",
        "인공지능이 감정을 가지게 된 미래를 배경으로 한 단편 소설을 써주세요.",
        "딥러닝과 머신러닝의 차이점을 설명해주세요.",
        "사용자: 주말에 서울에서 할 만한 활동을 추천해주세요.\\n어시스턴트:"
    ]
    
    # 추가 텍스트 데이터 (다양한 분야)
    contexts = [
        "인공지능(AI)은 인간의 학습능력과 추론능력, 지각능력, 자연언어의 이해능력 등을 컴퓨터 프로그램으로 실현한 기술이다.",
        "파이썬은 1991년 귀도 반 로섬이 발표한 고급 프로그래밍 언어로, 플랫폼 독립적이며 인터프리터식, 객체지향적, 동적 타이핑 대화형 언어이다.",
        "우주는 모든 시공간과 그 안에 존재하는 물질과 에너지, 우주 상수 등을, 총칭하여 부르는 말이다.",
        "양자역학은 미시 세계의 물리 현상을 설명하는 이론으로, 빛과 물질의 상호작용에 대한 기술을 제공한다.",
        "기후 변화는 지구의 기후 시스템이 전체적으로 변화하는 상태를 말하며, 기온, 강수량 같은 기후 요소의 평균값이 변화하는 현상을 포함한다."
    ]
    
    # 캘리브레이션 데이터 생성
    calibration_data = []
    
    # 무작위 샘플 생성
    for _ in range(num_samples):
        if random.random() < 0.7:
            # 프롬프트 기반 샘플
            text = random.choice(prompts)
            # 길이를 늘리기 위해 때때로 컨텍스트 추가
            if random.random() < 0.3:
                text = random.choice(contexts) + "\\n\\n" + text
        else:
            # 컨텍스트 기반 샘플
            text = random.choice(contexts)
            # 때때로 두 컨텍스트 결합
            if random.random() < 0.3:
                text += "\\n\\n" + random.choice(contexts)
        
        # GPTQ 형식으로 저장 (텍스트 + 샘플 ID)
        calibration_data.append({"text": text, "id": f"sample_{len(calibration_data)}"})
    
    # 파일에 저장
    with open("gptq_calibration_data.jsonl", "w", encoding="utf-8") as f:
        for item in calibration_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\\n")
    
    print(f"GPTQ 캘리브레이션 데이터 생성 완료: {len(calibration_data)} 샘플")

# 메인 함수 실행
generate_gptq_calibration_data()
'
else
  echo "기존 gptq_calibration_data.jsonl 파일을 사용합니다."
fi
""")

            # 양자화 스크립트 생성
            f.write("""# 4. GPTQ 양자화 실행
echo "GPTQ 양자화 실행 중..."

python -c '
import torch
import gc
import json
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

# 설정 로드
with open("gptq_config.json", "r") as f:
    config = json.load(f)

model_name = "${MODEL_NAME}"
output_dir = "model_gptq_quantized"

print(f"모델 {model_name}을 GPTQ로 양자화합니다.")
print(f"양자화 설정: {config}")

# 토크나이저 로드
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
except Exception as e:
    print(f"토크나이저 로드 중 오류: {e}")
    tokenizer = None

# 캘리브레이션 데이터 로드
texts = []
with open("gptq_calibration_data.jsonl", "r") as f:
    for line in f:
        data = json.loads(line)
        texts.append(data["text"])
print(f"캘리브레이션에 {len(texts)} 샘플을 사용합니다.")

# 양자화 설정 구성
quantize_config = BaseQuantizeConfig(
    bits=config.get("bits", 4),
    group_size=config.get("group_size", 128),
    desc_act=config.get("desc_act", True),
    sym=config.get("sym", True),
    true_sequential=config.get("true_sequential", True)
)

try:
    # 메모리 정리
    gc.collect()
    torch.cuda.empty_cache()
    
    # 모델 로드
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    # 양자화 모델 준비
    gptq_model = AutoGPTQForCausalLM.from_pretrained(
        model, 
        quantize_config=quantize_config,
        low_cpu_mem_usage=True,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    # 양자화 수행
    print("양자화 시작...")
    gptq_model.quantize(
        examples=[
            tokenizer(
                text, 
                return_tensors="pt",
                max_length=512,
                truncation=True
            ) for text in texts[:32]  # 메모리 한계로 일부만 사용
        ]
    )
    
    # 모델 저장
    print("양자화된 모델 저장 중...")
    gptq_model.save_quantized(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # 설정 저장
    with open(f"{output_dir}/quantize_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("GPTQ 양자화 및 저장 완료!")
except Exception as e:
    print(f"양자화 중 오류: {e}")
    import traceback
    traceback.print_exc()
'

# 양자화가 성공했는지 확인
if [ -d "model_gptq_quantized" ]; then
  echo "양자화가 성공적으로 완료되었습니다!"
  echo "양자화된 모델 위치: model_gptq_quantized"
else
  echo "양자화에 실패했습니다."
fi
""")
            
            # 테스트 스크립트 추가
            f.write("""# 5. 양자화된 모델 테스트
echo "양자화된 모델 테스트 중..."

python -c '
import torch
from auto_gptq import AutoGPTQForCausalLM
from transformers import AutoTokenizer

# 양자화된 모델 로드
model_path = "model_gptq_quantized"
try:
    # 모델 및 토크나이저 로드
    model = AutoGPTQForCausalLM.from_quantized(
        model_path,
        device_map="auto",
        use_safetensors=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # 간단한 추론 테스트
    prompt = "안녕하세요? 저는 GPTQ로 양자화된 모델입니다. 간단한 소개를 해볼게요:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # 생성
    print("생성 시작...")
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.7,
            top_p=0.9
        )
    
    # 결과 출력
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    print("\\n출력 결과:", response)
    print("GPTQ 양자화 모델 테스트 완료!")
except Exception as e:
    print(f"모델 테스트 중 오류: {e}")
    import traceback
    traceback.print_exc()
'
""")
            
        # 실행 권한 추가
        os.chmod(output_path, 0o755)
        
        return output_path