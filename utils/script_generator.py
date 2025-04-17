#!/usr/bin/env python3
# utils/script_generator.py - 양자화 스크립트 생성 도구

import os
import json
from typing import Dict, Any, Optional

def generate_quantization_script(
    model_name: str, 
    quantization_plan: Dict[str, Any], 
    method: str = "llamacpp",
    output_dir: str = "./output"
) -> str:
    """
    양자화 스크립트 생성
    
    Args:
        model_name (str): 모델 이름
        quantization_plan (Dict[str, Any]): 양자화 계획
        method (str): 양자화 방식 (llamacpp, awq, gptq)
        output_dir (str): 출력 디렉토리
    
    Returns:
        str: 생성된 스크립트 경로
    """
    # 방식에 따라 적절한 스크립트 생성 함수 호출
    if method == "llamacpp":
        return generate_llamacpp_script(model_name, quantization_plan, output_dir)
    elif method == "awq":
        return generate_awq_script(model_name, quantization_plan, output_dir)
    elif method == "gptq":
        return generate_gptq_script(model_name, quantization_plan, output_dir)
    else:
        raise ValueError(f"지원하지 않는 양자화 방식: {method}")

def generate_llamacpp_script(
    model_name: str, 
    quantization_plan: Dict[str, Any], 
    output_dir: str = "./output"
) -> str:
    """
    llama.cpp 양자화 스크립트 생성
    
    Args:
        model_name (str): 모델 이름
        quantization_plan (Dict[str, Any]): 양자화 계획
        output_dir (str): 출력 디렉토리
    
    Returns:
        str: 생성된 스크립트 경로
    """
    # 스크립트 저장 경로
    quant_dir = os.path.join(output_dir, "quantization_plan")
    os.makedirs(quant_dir, exist_ok=True)
    script_path = os.path.join(quant_dir, "quantization_command.sh")
    
    with open(script_path, 'w') as f:
        f.write("#!/bin/bash\n\n")
        f.write("# 이 스크립트는 모델을 llama.cpp로 양자화하기 위한 명령어입니다.\n\n")

        f.write("set -e\n")

        # 모델명 변수 추가
        f.write(f"MODEL_NAME=\"{model_name}\"\n")
        f.write("IFS='/' read -ra MODEL_PARTS <<< \"$MODEL_NAME\"\n")
        f.write("MODEL_PATH=~/.cache/huggingface/hub/models--${MODEL_PARTS[0]}--${MODEL_PARTS[1]}/snapshots/\n")
        f.write("DIR_NAME=`ls $MODEL_PATH`\n\n")
        
        # 캘리브레이션 데이터 확인
        f.write("# 0. 캘리브레이션 데이터 확인\n")
        f.write("if [ ! -f \"../calibration_data/calibration_data.txt\" ]; then\n")
        f.write("  echo \"캘리브레이션 데이터가 없습니다. 기본 데이터를 생성합니다.\"\n")
        f.write("  # 여기에 기본 캘리브레이션 데이터 생성 코드 추가\n")
        f.write("else\n")
        f.write("  echo \"기존 캘리브레이션 데이터를 사용합니다.\"\n")
        f.write("  cp ../calibration_data/calibration_data.txt .\n")
        f.write("fi\n\n")
        
        # llama.cpp 컴파일
        f.write("# 1. llama.cpp 컴파일 (GPU 지원 활성화)\n")
        f.write("if [ ! -d \"llama.cpp\" ]; then\n")
        f.write("  git clone https://github.com/ggerganov/llama.cpp\n")
        f.write("fi\n")
        f.write("cd llama.cpp\n")
        f.write("mkdir -p build\n")
        f.write("cd build\n")
        f.write("cmake .. -DGGML_CUDA=ON -DLLAMA_BUILD_TOOLS=ON\n")
        f.write("make -j$(nproc)\n")
        f.write("cd ../..\n\n")
        
        # GGUF 변환
        f.write("# 2. 모델을 GGUF 형식으로 변환\n")
        f.write("if [ ! -f \"model_fp16.gguf\" ]; then\n")
        f.write("  python llama.cpp/convert_hf_to_gguf.py \"$MODEL_PATH/$DIR_NAME\" --outtype f16 --outfile model_fp16.gguf\n")
        f.write("fi\n\n")
        
        # imatrix 생성
        f.write("# 3. 중요도 행렬(imatrix) 생성\n")
        f.write("if [ ! -f \"model.imatrix\" ]; then\n")
        f.write("  echo \"중요도 행렬(imatrix) 생성 중...\"\n")
        f.write("  if [ -f \"llama.cpp/build/bin/llama-imatrix\" ]; then\n")
        f.write("    # llama-imatrix 도구를 사용하여 imatrix 생성\n")
        f.write("    ./llama.cpp/build/bin/llama-imatrix -m model_fp16.gguf -f calibration_data.txt -o model.imatrix -ngl 99\n")
        f.write("  else\n")
        f.write("    # llama-imatrix가 없으면 llama-calibrate 사용\n")
        f.write("    ./llama.cpp/build/bin/llama-calibrate model_fp16.gguf --imatrix-file model.imatrix --imatrix-use calibration_data.txt\n")
        f.write("  fi\n")
        f.write("fi\n\n")
        
        # 양자화 타입 확인
        quantization_type = quantization_plan.get("quantization_type", "q4_k")
        f.write("# 양자화 타입 목록 확인\n")
        f.write("./llama.cpp/build/bin/llama-quantize --help | grep -i \"available finetypes\"\n\n")
        
        # 양자화 실행
        f.write("# 4. 양자화 실행\n")
        f.write("echo \"모델 양자화 실행 중...\"\n")
        f.write(f"./llama.cpp/build/bin/llama-quantize \\\n")
        f.write("  $([ -f \"model.imatrix\" ] && echo \"--imatrix model.imatrix\") \\\n")
        f.write("  model_fp16.gguf \\\n")
        f.write(f"  model_dynamic_{quantization_type}.gguf \\\n")
        f.write(f"  {quantization_type}\n\n")
        
        # 모델 테스트
        f.write("# 5. 모델 테스트\n")
        f.write("echo \"양자화된 모델 테스트 중...\"\n")
        f.write("./llama.cpp/build/bin/llama-cli \\\n")
        f.write(f"  -m model_dynamic_{quantization_type}.gguf \\\n")
        f.write("  -t 8 \\\n")
        f.write("  -n 128 \\\n")
        f.write("  -ngl 1 \\\n")
        f.write("  -p \"안녕하세요? 저는 양자화된 모델입니다. 간단한 소개를 해볼게요:\" \\\n")
        f.write("  --quiet\n\n")
        
        # 완료 메시지
        f.write("# 6. 완료 메시지\n")
        f.write("echo \"양자화가 성공적으로 완료되었습니다!\"\n")
        f.write(f"echo \"양자화된 모델 파일: model_dynamic_{quantization_type}.gguf\"\n")
    
    # 실행 권한 추가
    os.chmod(script_path, 0o755)
    
    return script_path

def generate_awq_script(
    model_name: str, 
    quantization_plan: Dict[str, Any], 
    output_dir: str = "./output"
) -> str:
    """
    AWQ 양자화 스크립트 생성
    
    Args:
        model_name (str): 모델 이름
        quantization_plan (Dict[str, Any]): 양자화 계획
        output_dir (str): 출력 디렉토리
    
    Returns:
        str: 생성된 스크립트 경로
    """
    # 스크립트 저장 경로
    script_dir = os.path.join(output_dir, "scripts")
    os.makedirs(script_dir, exist_ok=True)
    script_path = os.path.join(script_dir, "awq_quantization.sh")
    
    # 설정값 추출
    w_bit = quantization_plan.get("bits", 4)
    q_group_size = quantization_plan.get("group_size", 128)
    zero_point = quantization_plan.get("zero_point", True)
    
    with open(script_path, 'w') as f:
        f.write("#!/bin/bash\n\n")
        f.write("# 이 스크립트는 모델을 AWQ로 양자화하기 위한 명령어입니다.\n\n")

        f.write("set -e\n")

        # 모델명 변수 추가
        f.write(f"MODEL_NAME=\"{model_name}\"\n\n")
        
        # AWQ 패키지 설치
        f.write("# 1. AutoAWQ 패키지 설치 (필요한 경우)\n")
        f.write("pip install -U autoawq packaging auto-gptq\n\n")
        
        # 캘리브레이션 데이터 확인
        f.write("# 2. 캘리브레이션 데이터 확인\n")
        f.write("if [ ! -f \"../calibration_data/awq_calibration_data.jsonl\" ]; then\n")
        f.write("  echo \"AWQ 캘리브레이션 데이터가 없습니다. 기본 데이터를 생성합니다.\"\n")
        f.write("""  python -c '
import json
import random
import os

def generate_awq_calibration_data(num_samples=128, tokens_per_sample=384):
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
    
    # 캘리브레이션 데이터
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
        
        # AWQ 형식으로 저장 (원시 텍스트만 필요)
        calibration_data.append({"text": text})
    
    # 파일에 저장
    with open("awq_calibration_data.jsonl", "w", encoding="utf-8") as f:
        for item in calibration_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\\n")
    
    print(f"AWQ 캘리브레이션 데이터 생성 완료: {len(calibration_data)} 샘플")

# 메인 함수 실행
generate_awq_calibration_data()
'""")
        f.write("else\n")
        f.write("  echo \"기존 AWQ 캘리브레이션 데이터를 사용합니다.\"\n")
        f.write("  cp ../calibration_data/awq_calibration_data.jsonl .\n")
        f.write("fi\n\n")
        
        # 양자화 실행
        f.write("# 3. AWQ 양자화 실행\n")
        f.write("echo \"AWQ 양자화 실행 중...\"\n")
        f.write("""python -c '
import torch
import gc
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
import json
import os

# 설정 로드
model_name = "${MODEL_NAME}"
output_dir = "model_awq_quantized"
w_bit = """ + str(w_bit) + """
q_group_size = """ + str(q_group_size) + """
zero_point = """ + str(zero_point).lower() + """

print(f"모델 {model_name}을 AWQ로 양자화합니다.")
print(f"설정: w_bit={w_bit}, q_group_size={q_group_size}, zero_point={zero_point}")

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
with open("awq_calibration_data.jsonl", "r") as f:
    for line in f:
        texts.append(json.loads(line)["text"])
print(f"캘리브레이션에 {len(texts)} 샘플을 사용합니다.")

# 모델 로드 및 양자화
try:
    # 메모리 정리
    gc.collect()
    torch.cuda.empty_cache()
    
    # AWQ 모델 초기화
    model = AutoAWQForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16
    )
    
    # 양자화 수행
    model.quantize(
        tokenizer=tokenizer,
        quant_config={
            "zero_point": zero_point,
            "q_group_size": q_group_size,
            "w_bit": w_bit,
            "version": "gemm"
        },
        calib_data=texts
    )
    
    # 모델 저장
    model.save_quantized(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("AWQ 양자화 및 저장 완료!")
except Exception as e:
    print(f"양자화 중 오류: {e}")
    import traceback
    traceback.print_exc()
'""")
        
        # 양자화 완료 확인
        f.write("""
# 양자화가 성공했는지 확인
if [ -d "model_awq_quantized" ]; then
  echo "양자화가 성공적으로 완료되었습니다!"
  echo "양자화된 모델 위치: model_awq_quantized"
  
  # 전체 출력 디렉토리로 복사
  mkdir -p ../quantized_model/awq
  cp -r model_awq_quantized/* ../quantized_model/awq/
  echo "최종 모델이 ../quantized_model/awq/ 디렉토리에 복사되었습니다."
else
  echo "양자화에 실패했습니다."
fi
""")
        
        # 테스트 실행
        f.write("""
# 4. 양자화된 모델 테스트
echo "양자화된 모델 테스트 중..."

python -c '
import torch
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

# 양자화된 모델 로드
model_path = "model_awq_quantized"
try:
    # 모델 및 토크나이저 로드
    model = AutoAWQForCausalLM.from_quantized(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # 간단한 추론 테스트
    prompt = "안녕하세요? 저는 AWQ로 양자화된 모델입니다. 간단한 소개를 해볼게요:"
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
    print("AWQ 양자화 모델 테스트 완료!")
except Exception as e:
    print(f"모델 테스트 중 오류: {e}")
    import traceback
    traceback.print_exc()
'
""")
    
    # 실행 권한 추가
    os.chmod(script_path, 0o755)
    
    return script_path

def generate_gptq_script(
    model_name: str, 
    quantization_plan: Dict[str, Any], 
    output_dir: str = "./output"
) -> str:
    """
    GPTQ 양자화 스크립트 생성
    
    Args:
        model_name (str): 모델 이름
        quantization_plan (Dict[str, Any]): 양자화 계획
        output_dir (str): 출력 디렉토리
    
    Returns:
        str: 생성된 스크립트 경로
    """
    # 스크립트 저장 경로
    script_dir = os.path.join(output_dir, "scripts")
    os.makedirs(script_dir, exist_ok=True)
    script_path = os.path.join(script_dir, "gptq_quantization.sh")
    
    # 설정값 추출
    bits = quantization_plan.get("bits", 4)
    group_size = quantization_plan.get("group_size", 128)
    act_order = quantization_plan.get("act_order", True)
    
    # 설정 내용 JSON으로 만들기
    gptq_config = {
        "method": "gptq",
        "bits": bits,
        "group_size": group_size,
        "act_order": act_order,
        "desc_act": True,
        "sym": True,
        "true_sequential": True,
        "calibration_needed": True
    }
    
    with open(script_path, 'w') as f:
        f.write("#!/bin/bash\n\n")
        f.write("# 이 스크립트는 모델을 GPTQ로 양자화하기 위한 명령어입니다.\n\n")

        f.write("set -e\n\n")

        # 모델명 변수 추가
        f.write(f"MODEL_NAME=\"{model_name}\"\n\n")
        
        # GPTQ 패키지 설치
        f.write("# 1. Auto-GPTQ 패키지 설치 (필요한 경우)\n")
        f.write("pip install -U auto-gptq packaging optimum\n\n")
        
        # 양자화 설정 저장
        f.write("# 2. 양자화 설정 저장\n")
        f.write("cat > gptq_config.json << 'EOL'\n")
        f.write(json.dumps(gptq_config, indent=2))
        f.write("\nEOL\n\n")
        
        # 캘리브레이션 데이터 확인
        f.write("# 3. 캘리브레이션 데이터 확인\n")
        f.write("if [ ! -f \"../calibration_data/gptq_calibration_data.jsonl\" ]; then\n")
        f.write("  echo \"GPTQ 캘리브레이션 데이터가 없습니다. 기본 데이터를 생성합니다.\"\n")
        f.write("""  python -c '
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
'""")
        f.write("else\n")
        f.write("  echo \"기존 GPTQ 캘리브레이션 데이터를 사용합니다.\"\n")
        f.write("  cp ../calibration_data/gptq_calibration_data.jsonl .\n")
        f.write("fi\n\n")
        
        # 양자화 실행
        f.write("# 4. GPTQ 양자화 실행\n")
        f.write("echo \"GPTQ 양자화 실행 중...\"\n")
        f.write("""python -c '
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
    act_order=config.get("act_order", True),
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
'""")

        # 양자화 완료 확인
        f.write("""
# 양자화가 성공했는지 확인
if [ -d "model_gptq_quantized" ]; then
  echo "양자화가 성공적으로 완료되었습니다!"
  echo "양자화된 모델 위치: model_gptq_quantized"
  
  # 전체 출력 디렉토리로 복사
  mkdir -p ../quantized_model/gptq
  cp -r model_gptq_quantized/* ../quantized_model/gptq/
  echo "최종 모델이 ../quantized_model/gptq/ 디렉토리에 복사되었습니다."
else
  echo "양자화에 실패했습니다."
fi
""")

        # 테스트 실행
        f.write("""
# 5. 양자화된 모델 테스트
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
    os.chmod(script_path, 0o755)
    
    return script_path

def generate_repetition_prevention_script(output_dir: str) -> str:
    """
    반복 출력 방지 스크립트 생성 (llama.cpp 모델용)
    
    Args:
        output_dir (str): 출력 디렉토리
    
    Returns:
        str: 생성된 스크립트 경로
    """
    # 스크립트 저장 경로
    quant_dir = os.path.join(output_dir, "quantization_plan")
    os.makedirs(quant_dir, exist_ok=True)
    script_path = os.path.join(quant_dir, "prevent_repetition.sh")
    
    with open(script_path, 'w') as f:
        f.write("#!/bin/bash\n\n")
        f.write("# 이 스크립트는 llama.cpp 모델의 반복 출력을 방지하기 위한 설정을 제공합니다.\n\n")
        
        f.write("# 모델 파일 찾기\n")
        f.write("MODEL_FILE=$(find . -name \"*.gguf\" | grep -v \"fp16\" | head -n 1)\n\n")
        
        f.write("if [ -z \"$MODEL_FILE\" ]; then\n")
        f.write("  echo \"양자화된 모델 파일을 찾을 수 없습니다.\"\n")
        f.write("  exit 1\n")
        f.write("fi\n\n")
        
        f.write("echo \"모델 파일: $MODEL_FILE\"\n\n")
        
        f.write("# 반복 방지 설정으로 모델 실행\n")
        f.write("echo \"반복 방지 설정으로 모델 실행 중...\"\n")
        f.write("./llama.cpp/build/bin/llama-cli \\\n")
        f.write("  -m \"$MODEL_FILE\" \\\n")
        f.write("  -t 8 \\\n")
        f.write("  -n 512 \\\n")
        f.write("  -ngl 1 \\\n")
        f.write("  --repeat-penalty 1.3 \\\n")
        f.write("  --repeat-last-n 64 \\\n")
        f.write("  --temp 0.7 \\\n")
        f.write("  --top-p 0.9 \\\n")
        f.write("  --top-k 40 \\\n")
        f.write("  -p \"사용자: 인공지능의 미래에 대해 간략하게 설명해주세요.\n어시스턴트: \"\n")
    
    # 실행 권한 추가
    os.chmod(script_path, 0o755)
    
    return script_path
