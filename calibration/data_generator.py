#!/usr/bin/env python3
# data_generator.py - 캘리브레이션 데이터 생성

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import json
from tqdm import tqdm
import time
import random

class CalibrationDataGenerator:
    """모델 특화 캘리브레이션 데이터 생성기"""
    
    def __init__(self, model_name, output_dir="calibration_data", device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        캘리브레이션 데이터 생성기 초기화
        
        Args:
            model_name (str): 모델 이름
            output_dir (str): 결과 저장 디렉토리
            device (str): 사용할 디바이스
        """
        self.model_name = model_name
        self.output_dir = output_dir
        self.device = device
        self.tokenizer = None
        self.model = None
        
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"CalibrationDataGenerator 초기화: {model_name}")
        print(f"디바이스: {device}")
    
    def load_model(self):
        """모델 및 토크나이저 로드"""
        print(f"\n모델 로드 중: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map=self.device
        )
        
        print(f"모델 로드 완료: {type(self.model).__name__}")
        return self.model
    
    def generate_prompts(self, num_prompts=50):
        """다양한 유형의 프롬프트 생성
        
        Args:
            num_prompts (int): 생성할 프롬프트 수
            
        Returns:
            list: 생성된 프롬프트 목록
        """
        prompts = []
        
        # 프롬프트 카테고리 및 템플릿
        categories = {
            "한국어 일반": [
                "인공지능의 미래에 대해 설명해주세요.",
                "기후 변화의 주요 원인과 해결책은 무엇인가요?",
                "한국 역사의 중요한 사건들을 시간순으로 나열해주세요.",
                "메타버스란 무엇이며 어떤 가능성을 가지고 있나요?",
                "현대 사회에서 개인정보 보호의 중요성에 대해 논의해주세요.",
                "인간과 AI의 관계는 앞으로 어떻게 발전할까요?",
                "블록체인 기술의 활용 사례를 설명해주세요.",
                "지속 가능한 발전이란 무엇이며 왜 중요한가요?",
                "한국 문화의 특징과 세계적 인기의 이유는 무엇인가요?",
                "현대 교육의 문제점과 개선 방안에 대해 논의해주세요."
            ],
            "코드 생성": [
                "파이썬으로 간단한 웹 크롤러를 만들어주세요.",
                "자바스크립트로 투두리스트 앱을 만드는 코드를 작성해주세요.",
                "파이썬으로 이진 탐색 알고리즘을 구현해주세요.",
                "SQL로 직원과 부서 정보를 조회하는 쿼리를 작성해주세요.",
                "파이썬으로 머신러닝 모델을 학습하고 예측하는 코드를 작성해주세요.",
                "HTML과 CSS로 반응형 네비게이션 바를 만들어주세요.",
                "파이썬으로 이미지 처리 스크립트를 작성해주세요.",
                "자바로 간단한 채팅 서버를 구현해주세요.",
                "파이썬으로 CSV 파일을 처리하는 코드를 작성해주세요.",
                "React로 날씨 정보를 보여주는 컴포넌트를 만들어주세요."
            ],
            "창의적 작업": [
                "우주 탐험가의 일기를 작성해주세요.",
                "인공지능이 감정을 가지게 된 미래를 배경으로 한 단편 소설을 써주세요.",
                "환경 보호를 주제로 한 시를 작성해주세요.",
                "가상 인터뷰: 과학자와 예술가의 대화를 작성해주세요.",
                "지구를 방문한 외계인의 관점에서 인간 사회를 묘사해주세요.",
                "블록체인이 일상생활을 완전히 바꾼 2050년 세계를 상상해 설명해주세요.",
                "디스토피아 세계관의 소설 첫 장을 작성해주세요.",
                "동화: '용기 있는 AI 로봇의 모험'을 써주세요.",
                "가상의 신제품 출시 연설문을 작성해주세요.",
                "타임머신을 타고 과거로 여행한 사람의 경험을 담은 편지를 작성해주세요."
            ],
            "전문 지식": [
                "양자 컴퓨팅의 기본 원리와 응용 분야를 설명해주세요.",
                "딥러닝과 머신러닝의 차이점을 설명해주세요.",
                "블록체인의 작동 방식과 핵심 기술을 설명해주세요.",
                "신경망의 종류와 각각의 장단점을 비교해주세요.",
                "강화학습의 기본 개념과, Q-러닝이 어떻게 작동하는지 설명해주세요.",
                "대규모 언어 모델의 발전 과정과 주요 모델들을 설명해주세요.",
                "자연어 처리의 주요 과제와, 최신 기술이 이를 어떻게 해결하는지 설명해주세요.",
                "컴퓨터 비전 분야의 주요 알고리즘과 응용 사례를 설명해주세요.",
                "생성형 AI의 작동 원리와 최근 발전 동향을 설명해주세요.",
                "연합 학습(Federated Learning)이란 무엇이며, 왜 중요한가요?"
            ],
            "대화형": [
                "사용자: 안녕하세요, 오늘 기분이 어떠세요?\n어시스턴트:",
                "사용자: 주말에 서울에서 할 만한 활동을 추천해주세요.\n어시스턴트:",
                "사용자: 프로그래밍을 처음 배우려면 어떤 언어부터 시작하는 게 좋을까요?\n어시스턴트:",
                "사용자: 영어 공부에 효과적인 방법을 알려주세요.\n어시스턴트:",
                "사용자: 요즘 읽을만한 좋은 책을 추천해주세요.\n어시스턴트:",
                "사용자: 건강한 식습관을 유지하는 팁을 알려주세요.\n어시스턴트:",
                "사용자: 인공지능이 가진 위험성은 무엇인가요?\n어시스턴트:",
                "사용자: 스트레스를 해소하는 방법을 알려주세요.\n어시스턴트:",
                "사용자: 직장에서 갈등 상황을 해결하는 좋은 방법이 있을까요?\n어시스턴트:",
                "사용자: 지속 가능한 생활 방식으로 전환하는 팁을 주세요.\n어시스턴트:"
            ]
        }
        
        # 각 카테고리에서 프롬프트 선택
        for category, templates in categories.items():
            # 카테고리별 비율 조정 (필요에 따라)
            category_count = max(1, int(num_prompts * (1 / len(categories))))
            
            # 랜덤 선택 (중복 허용)
            for _ in range(category_count):
                prompt = random.choice(templates)
                prompts.append(prompt)
        
        # 충분한 수의 프롬프트가 생성되지 않은 경우 추가
        while len(prompts) < num_prompts:
            category = random.choice(list(categories.keys()))
            prompt = random.choice(categories[category])
            prompts.append(prompt)
        
        # 최종 프롬프트 목록을 랜덤하게 섞음
        random.shuffle(prompts)
        
        return prompts[:num_prompts]
    
    def generate_responses(self, prompts, max_tokens=512):
        """모델을 사용하여 프롬프트에 대한 응답 생성
        
        Args:
            prompts (list): 프롬프트 목록
            max_tokens (int): 생성할 최대 토큰 수
            
        Returns:
            list: 프롬프트-응답 쌍 목록
        """
        if self.model is None:
            self.load_model()
        
        responses = []
        
        for prompt in tqdm(prompts, desc="응답 생성"):
            try:
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                
                # 생성 옵션 설정
                gen_config = {
                    "max_new_tokens": max_tokens,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "do_sample": True
                }
                
                # 응답 생성
                with torch.no_grad():
                    output_ids = self.model.generate(
                        inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        **gen_config
                    )
                
                # 입력 토큰 제외하고 출력 디코딩
                input_length = inputs.input_ids.shape[1]
                response = self.tokenizer.decode(output_ids[0][input_length:], skip_special_tokens=True)
                
                # 프롬프트-응답 쌍 저장
                pair = {
                    "prompt": prompt,
                    "response": response
                }
                responses.append(pair)
                
                # 잠시 대기 (모델 과부하 방지)
                time.sleep(0.5)
                
            except Exception as e:
                print(f"오류 발생: {str(e)}")
                continue
        
        return responses
    
    def create_calibration_file(self, responses, include_prompts=True):
        """캘리브레이션 데이터 파일 생성
        
        Args:
            responses (list): 프롬프트-응답 쌍 목록
            include_prompts (bool): 프롬프트 포함 여부
            
        Returns:
            str: 생성된 캘리브레이션 파일 경로
        """
        calibration_text = ""
        
        # 프롬프트와 응답을 텍스트로 결합
        for pair in responses:
            if include_prompts:
                calibration_text += pair["prompt"] + "\n\n"
            calibration_text += pair["response"] + "\n\n"
            calibration_text += "-" * 40 + "\n\n"  # 구분자 추가
        
        # 파일 저장
        output_path = os.path.join(self.output_dir, "calibration_data.txt")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(calibration_text)
        
        # JSON 형식으로도 저장 (추후 분석용)
        json_path = os.path.join(self.output_dir, "calibration_pairs.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(responses, f, ensure_ascii=False, indent=2)
        
        print(f"캘리브레이션 데이터 파일 생성 완료: {output_path}")
        print(f"JSON 형식 데이터 파일 생성 완료: {json_path}")
        
        return output_path
    
    def add_to_shell_script(self, calibration_path):
        """양자화 쉘 스크립트에 캘리브레이션 데이터 사용 코드 추가
        
        Args:
            calibration_path (str): 캘리브레이션 파일 경로
            
        Returns:
            str: 생성된 스크립트 경로
        """
        script_path = os.path.join(self.output_dir, "use_calibration_data.sh")
        
        with open(script_path, "w") as f:
            f.write("#!/bin/bash\n\n")
            f.write("# 이 스크립트는 자동 생성된 캘리브레이션 데이터를 양자화에 사용하는 예시입니다.\n\n")
            
            f.write(f"CALIBRATION_FILE=\"{calibration_path}\"\n\n")
            
            f.write("# 캘리브레이션 파일 확인\n")
            f.write("if [ ! -f \"$CALIBRATION_FILE\" ]; then\n")
            f.write("  echo \"오류: 캘리브레이션 파일을 찾을 수 없습니다: $CALIBRATION_FILE\"\n")
            f.write("  exit 1\n")
            f.write("fi\n\n")
            
            f.write("echo \"캘리브레이션 데이터 정보:\"\n")
            f.write("echo \"  - 파일: $CALIBRATION_FILE\"\n")
            f.write("echo \"  - 크기: $(du -h \"$CALIBRATION_FILE\" | cut -f1)\"\n")
            f.write("echo \"  - 라인 수: $(wc -l < \"$CALIBRATION_FILE\")\"\n")
            f.write("echo \"  - 단어 수: $(wc -w < \"$CALIBRATION_FILE\")\"\n\n")
            
            f.write("# llama.cpp의 중요도 행렬(imatrix) 생성 명령어 예시\n")
            f.write("echo \"중요도 행렬 생성 명령어 예시:\"\n")
            f.write("echo \"./llama.cpp/build/bin/llama-imatrix -m model_fp16.gguf -f \\\"$CALIBRATION_FILE\\\" -o model.imatrix -ngl 99\"\n")
        
        # 실행 권한 부여
        os.chmod(script_path, 0o755)
        
        print(f"캘리브레이션 데이터 사용 스크립트 생성 완료: {script_path}")
        return script_path
    
    def run_generation_pipeline(self, num_prompts=50, max_tokens=512):
        """캘리브레이션 데이터 생성 파이프라인 실행
        
        Args:
            num_prompts (int): 생성할 프롬프트 수
            max_tokens (int): 생성할 최대 토큰 수
            
        Returns:
            dict: 생성 결과 정보
        """
        start_time = time.time()
        
        print(f"\n== 모델 특화 캘리브레이션 데이터 생성 파이프라인 시작 ==")
        print(f"모델: {self.model_name}")
        print(f"생성할 프롬프트 수: {num_prompts}")
        
        # 1. 프롬프트 생성
        prompts = self.generate_prompts(num_prompts)
        print(f"프롬프트 {len(prompts)}개 생성 완료")
        
        # 2. 응답 생성
        responses = self.generate_responses(prompts, max_tokens)
        print(f"응답 {len(responses)}개 생성 완료")
        
        # 3. 캘리브레이션 데이터 파일 생성
        calibration_path = self.create_calibration_file(responses)
        
        # 4. 쉘 스크립트에 추가
        script_path = self.add_to_shell_script(calibration_path)
        
        elapsed_time = time.time() - start_time
        print(f"\n== 캘리브레이션 데이터 생성 완료 ==")
        print(f"총 소요 시간: {elapsed_time:.2f}초 ({elapsed_time/60:.2f}분)")
        
        return {
            "calibration_path": calibration_path,
            "script_path": script_path,
            "num_responses": len(responses),
            "elapsed_time": elapsed_time
        }