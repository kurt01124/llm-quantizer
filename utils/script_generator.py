#!/usr/bin/env python3
# script_generator.py - 스크립트 생성 기능

import os

class ScriptGenerator:
    """스크립트 생성 유틸리티 클래스"""
    
    def __init__(self, model_name, output_dir="scripts"):
        """
        스크립트 생성기 초기화
        
        Args:
            model_name (str): 모델 이름
            output_dir (str): 출력 디렉토리
        """
        self.model_name = model_name
        self.output_dir = output_dir
        
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_quantization_script(self, super_weight_layers=None):
        """llama.cpp 양자화 명령어 스크립트 생성
        
        Args:
            super_weight_layers (list): 슈퍼 웨이트 레이어 목록 (선택 사항)
            
        Returns:
            str: 생성된 스크립트 경로
        """
        output_path = os.path.join(self.output_dir, "quantization_command.sh")
        
        with open(output_path, 'w') as f:
            f.write("#!/bin/bash\n\n")
            f.write("# 이 스크립트는 모델을 동적 양자화하기 위한 명령어입니다.\n\n")

            f.write("set -e\n")

            # 모델명 변수 추가
            f.write(f"MODEL_NAME=\"{self.model_name}\"\n")
            f.write("IFS='/' read -ra MODEL_PARTS <<< \"$MODEL_NAME\"\n")
            f.write("MODEL_PATH=~/.cache/huggingface/hub/models--${MODEL_PARTS[0]}--${MODEL_PARTS[1]}/snapshots/\n")
            f.write("DIR_NAME=`ls $MODEL_PATH`\n\n")
            
            # 캘리브레이션 데이터 생성 코드 추가
            f.write("""# 0. 모델 특화 calibration_data.txt 생성
echo "모델 특화 캘리브레이션 데이터 생성 중..."
if [ ! -f "calibration_data.txt" ]; then
  # 모델에서 자동 생성
  python -c '
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import random
import time
import os

def generate_model_specific_calibration():
    # 모델 정보
    model_name = "${MODEL_NAME}"
    print(f"모델 {model_name}에 대한 특화 캘리브레이션 데이터 생성 중...")
    
    # 다양한 프롬프트 준비
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
    
    try:
        # 토크나이저 및 모델 로드
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        calibration_text = ""
        
        # 각 프롬프트에 대한 응답 생성
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            # 생성 옵션 설정
            with torch.no_grad():
                output_ids = model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=512,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True
                )
            
            # 응답 디코딩 (입력 제외)
            input_length = inputs.input_ids.shape[1]
            response = tokenizer.decode(output_ids[0][input_length:], skip_special_tokens=True)
            
            # 프롬프트와 응답 추가
            calibration_text += prompt + "\\n\\n"
            calibration_text += response + "\\n\\n"
            calibration_text += "-" * 40 + "\\n\\n"  # 구분자
            
            # 모델 과부하 방지
            time.sleep(1)
        
        # 파일 저장
        with open("calibration_data.txt", "w", encoding="utf-8") as f:
            f.write(calibration_text)
        
        print(f"캘리브레이션 데이터 생성 완료: {os.path.getsize('calibration_data.txt')/1024:.1f} KB")
        return True
    
    except Exception as e:
        print(f"캘리브레이션 데이터 생성 중 오류 발생: {str(e)}")
        return False

# 메인 함수 실행
generate_model_specific_calibration()
'

  # 생성 실패한 경우 기존 방법으로 대체
  if [ ! -s "calibration_data.txt" ]; then
    echo "모델 특화 캘리브레이션 실패, 일반 캘리브레이션 데이터로 대체합니다."
    
    # 임시 디렉토리 생성
    mkdir -p temp_calibration
    cd temp_calibration
    
    # 1. 직접 몇 가지 샘플 텍스트 추가
    cat << 'EOF' > sample_text_ko.txt
인공지능(AI)은 인간의 지능을 모방하고 학습, 문제 해결, 패턴 인식 등을 수행할 수 있는 컴퓨터 시스템을 말합니다. 
현대 AI는 주로 딥러닝과 머신러닝 기술을 기반으로 하며, 자연어 처리, 컴퓨터 비전, 음성 인식 등 다양한 분야에 적용되고 있습니다.
인공지능의 발전은 최근 몇 년간 급속도로 진행되어 왔으며, 특히 대규모 언어 모델(LLM)과 같은 기술이 크게 발전했습니다.
이러한 기술들은 우리의 일상 생활부터 산업 분야까지 광범위하게 영향을 미치고 있습니다.
자연어 처리 기술은 언어 번역, 대화형 시스템, 내용 요약 등에 사용됩니다.
컴퓨터 비전 기술은 이미지 인식, 객체 감지, 얼굴 인식 등에 활용됩니다.
머신러닝 알고리즘은 데이터에서 패턴을 찾아 예측하거나 의사 결정을 내리는 데 도움을 줍니다.
딥러닝은 특히 복잡한 패턴을 인식하는 데 뛰어나며, 이미지 분류, 자연어 이해, 게임 플레이 등 다양한 작업에서 인간 수준의 성능을 달성했습니다.
인공지능 기술은 의료, 금융, 교육, 제조업 등 거의 모든 산업 분야에 혁신을 가져오고 있습니다.
한국은 인공지능 기술의 개발과 적용에 있어서 세계적으로 주목받는 국가 중 하나입니다.
인공지능의 윤리적 활용과 규제는 현대 사회의 중요한 화두 중 하나입니다.
데이터 프라이버시, 알고리즘 편향성, 자동화로 인한 일자리 변화 등 다양한 사회적 영향에 대한 논의가 활발히 이루어지고 있습니다.
EOF
    
    # 2. 영어 wikitext 샘플 다운로드 (약 20,000토큰)
    echo "영어 WikiText 데이터 다운로드 중..."
    curl -s -L "https://huggingface.co/datasets/ikawrakow/validation-datasets-for-llama.cpp/resolve/main/wiki.train.sample.txt" > wiki_en.txt || echo "위키텍스트 다운로드 실패"
    
    # 3. 프로그래밍 코드 샘플 (Python, JavaScript) - 약 10,000토큰
    echo "코드 샘플 다운로드 중..."
    curl -s "https://raw.githubusercontent.com/python/cpython/main/Lib/collections/abc.py" > code_python.txt || echo "Python 코드 다운로드 실패"
    curl -s "https://raw.githubusercontent.com/mrdoob/three.js/dev/src/math/Vector3.js" > code_js.txt || echo "JavaScript 코드 다운로드 실패"
    
    # 4. groups_merged-enhancedV3.txt의 샘플 다운로드 (약 10,000토큰)
    echo "groups_merged-enhancedV3 샘플 다운로드 중..."
    curl -s -L "https://huggingface.co/datasets/ikawrakow/validation-datasets-for-llama.cpp/resolve/main/groups_merged-enhancedV3.txt" | head -n 2000 > groups_sample.txt || echo "groups 샘플 다운로드 실패"
    
    # 5. 한국어 대화 예제 추가
    cat << 'EOF' > dialogue_ko.txt
사용자: 인공지능에 대해 설명해주세요.
어시스턴트: 인공지능(AI)은 인간의 지능을 모방하고 학습, 문제 해결, 패턴 인식 등을 수행할 수 있는 컴퓨터 시스템을 말합니다. 현대 AI는 주로 딥러닝과 머신러닝 기술을 기반으로 하며, 자연어 처리, 컴퓨터 비전, 음성 인식 등 다양한 분야에 적용되고 있습니다.

사용자: 파이썬으로 간단한 웹 크롤러를 만들려면 어떻게 해야 하나요?
어시스턴트: 파이썬으로 간단한 웹 크롤러를 만들기 위해서는 requests와 BeautifulSoup 라이브러리를 주로 사용합니다. 다음은 기본적인 예제입니다:

import requests
from bs4 import BeautifulSoup

# 웹 페이지 가져오기
url = "https://example.com"
response = requests.get(url)

# HTML 파싱
soup = BeautifulSoup(response.text, 'html.parser')

# 원하는 데이터 추출 (예: 모든 링크)
links = soup.find_all('a')
for link in links:
    print(link.get('href'))

이 코드를 실행하기 전에 먼저 필요한 라이브러리를 설치해야 합니다:
pip install requests beautifulsoup4
EOF
    
    # 모든 파일 합치기
    cat sample_text_ko.txt wiki_en.txt code_python.txt code_js.txt groups_sample.txt dialogue_ko.txt > ../calibration_data.txt
    
    # 다운로드 실패한 경우 기본 데이터 추가
    if [ ! -s ../calibration_data.txt ]; then
      echo "다운로드 실패, 기본 데이터만 사용합니다."
      cat sample_text_ko.txt dialogue_ko.txt > ../calibration_data.txt
    fi
    
    # 임시 디렉토리 정리
    cd ..
    rm -rf temp_calibration
    
    echo "캘리브레이션 데이터 생성 완료: $(wc -l < calibration_data.txt) 줄, $(wc -w < calibration_data.txt) 단어"
  fi
else
  echo "기존 calibration_data.txt 파일을 사용합니다."
fi
""")
            
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
            
            f.write("# 2. 모델을 GGUF 형식으로 변환\n")
            f.write("if [ ! -f \"model_fp16.gguf\" ]; then\n")
            f.write("  python llama.cpp/convert_hf_to_gguf.py \"$MODEL_PATH/$DIR_NAME\" --outtype f16 --outfile model_fp16.gguf\n")
            f.write("fi\n\n")
            
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
            
            f.write("# 양자화 타입 목록 확인\n")
            f.write("./llama.cpp/build/bin/llama-quantize --help | grep -i \"available finetypes\" || echo \"양자화 타입 목록을 확인할 수 없습니다. iq1_s를 사용합니다.\"\n\n")
            
            f.write("# 4. 양자화 실행 (표준 지원 타입 사용)\n")
            f.write("echo \"모델 양자화 실행 중...\"\n")

            f.write("./llama.cpp/build/bin/llama-quantize \\\n")
            f.write("  $([ -f \"model.imatrix\" ] && echo \"--imatrix model.imatrix\") \\\n")
            f.write("  model_fp16.gguf \\\n")
            f.write("  model_dynamic_iq1_s.gguf \\\n")
            f.write("  iq1_s \\\n")
            f.write("  4 \\\n")

            # 슈퍼 웨이트 레이어 오버라이드 설정
            if super_weight_layers and len(super_weight_layers) > 0:
                sw_patterns = "|".join(super_weight_layers)
                f.write(f"  --override-layer-type \"({sw_patterns}):q6_k\" \\\n")
            
            # 임베딩 및 출력 레이어
            f.write("  --override-layer-type \"*embed*.weight:q8_0\" \\\n")
            f.write("  --override-layer-type \"*lm_head*.weight:q8_0\" \\\n")
            
            # 초기 레이어
            f.write("  --override-layer-type \"layers.[0-3].*.weight:q6_k\" \\\n")
            
            # 어텐션 레이어
            f.write("  --override-layer-type \"*.attention.*.weight:q4_k\" \\\n")
            
            # 레이어 정규화 및 기타
            f.write("  --override-layer-type \"*.norm*.weight:f16\" \\\n")
            f.write("  --override-layer-type \"*.router*.weight:f16\"\n\n")
            
            f.write("# 5. 모델 테스트\n")
            f.write("if [ -f \"model_dynamic_iq1_s.gguf\" ]; then\n")
            f.write("  echo \"양자화된 모델 테스트:\"\n")
            f.write("  ./llama.cpp/build/bin/llama-cli \\\n")
            f.write("    --model model_dynamic_iq1_s.gguf \\\n")
            f.write("    --prompt \"인공지능의 미래에 대해 설명해주세요.\" \\\n")
            f.write("    --temp 0.7 \\\n")
            f.write("    --repeat-penalty 1.2 \\\n")
            f.write("    --ctx-size 2048\n")
            f.write("else\n")
            f.write("  echo \"양자화된 모델 파일을 찾을 수 없습니다.\"\n")
            f.write("fi\n")
        
        # 실행 권한 부여
        os.chmod(output_path, 0o755)
        
        print(f"\n양자화 명령어 스크립트가 생성되었습니다: {output_path}")
        return output_path
    
    def generate_repetition_prevention_script(self):
        """반복 방지 설정이 포함된 추론 스크립트 생성
        
        Returns:
            str: 생성된 스크립트 경로
        """
        output_path = os.path.join(self.output_dir, "prevent_repetition.sh")
        
        with open(output_path, 'w') as f:
            f.write("#!/bin/bash\n\n")
            f.write("# 이 스크립트는 양자화된 모델의 반복 출력을 방지하기 위한 설정을 포함합니다.\n\n")
            
            f.write("# 양자화된 모델 파일\n")
            f.write("MODEL_FILE=\"model_dynamic_iq1_s.gguf\"\n\n")
            
            f.write("# 모델 파일 확인\n")
            f.write("if [ ! -f \"$MODEL_FILE\" ]; then\n")
            f.write("  echo \"오류: 모델 파일을 찾을 수 없습니다: $MODEL_FILE\"\n")
            f.write("  exit 1\n")
            f.write("fi\n\n")
            
            f.write("# 반복 방지를 위한 추론 실행\n")
            f.write("./llama.cpp/build/bin/llama-cli \\\n")
            f.write("  --model \"$MODEL_FILE\" \\\n")
            f.write("  --prompt \"아래 주제에 대한 글을 작성해주세요:\\n\\n인공지능의 미래\" \\\n")
            f.write("  --temp 0.8 \\\n")
            f.write("  --top-p 0.9 \\\n")
            f.write("  --repeat-penalty 1.2 \\\n")
            f.write("  --repeat-last-n 128 \\\n")
            f.write("  --presence-penalty 0.2 \\\n")
            f.write("  --frequency-penalty 0.2 \\\n")
            f.write("  --mirostat 2 \\\n")
            f.write("  --mirostat-lr 0.1 \\\n")
            f.write("  --mirostat-ent 5.0 \\\n")
            f.write("  --ctx-size 2048 \\\n")
            f.write("  --n-predict 512\n\n")
            
            f.write("# 추가 반복 방지 옵션 설명:\n")
            f.write("# --temp 0.8: 온도. 낮을수록 결정적, 높을수록 무작위적.\n")
            f.write("# --top-p 0.9: Nucleus sampling. 확률 총합이 이 값 이상인 토큰만 고려.\n")
            f.write("# --repeat-penalty 1.2: 이미 생성된 토큰의 확률에 페널티 적용.\n")
            f.write("# --repeat-last-n 128: 페널티를 적용할 이전 토큰의 수.\n")
            f.write("# --presence-penalty 0.2: 이미 등장한 토큰에 페널티 적용.\n")
            f.write("# --frequency-penalty 0.2: 자주 등장한 토큰에 더 큰 페널티 적용.\n")
            f.write("# --mirostat 2: Mirostat 알고리즘 사용 (2 = 버전 2.0).\n")
            f.write("# --mirostat-lr 0.1: Mirostat 학습률.\n")
            f.write("# --mirostat-ent 5.0: 목표 엔트로피.\n")
        
        # 실행 권한 부여
        os.chmod(output_path, 0o755)
        
        print(f"\n반복 방지 스크립트가 생성되었습니다: {output_path}")
        return output_path