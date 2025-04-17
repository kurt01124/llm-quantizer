- # LLM 양자화 자동화 도구

  LLM(Large Language Model) 양자화를 자동화하는 도구입니다. 이 도구는 모델의 아키텍처를 분석하고, 레이어별 중요도를 평가하여 최적의 양자화 전략을 설계하고 구현합니다. llama.cpp(GGUF), AWQ, GPTQ 양자화 방식을 모두 지원합니다.

  ## 주요 기능

  1. **모델 아키텍처 분석**: 모델 구조와 레이어별 파라미터 수 분석
  2. **슈퍼 웨이트 탐지**: 중요한 가중치 식별 및 보존
  3. **동적 양자화 전략**: 레이어별 특성에 따른 최적의 비트 할당
  4. **캘리브레이션 데이터 생성**: 모델 특화 캘리브레이션 데이터 자동 생성
  5. **양자화 스크립트 생성**: 다양한 양자화 방식(llama.cpp, AWQ, GPTQ)에 대한 자동화 스크립트
  6. **모듈화된 설계**: 새로운 양자화 방식을 쉽게 추가할 수 있는 확장 가능한 구조
  7. **반복 출력 방지**: 양자화 모델의 반복 출력 방지 설정

  ## 설치 방법

  ```bash
  # 저장소 클론
  git clone https://github.com/kurtz01124/llm-quantizer.git
  cd llm-quantizer
  
  # 의존성 설치
  pip install -r requirements.txt
  
  # CUDA 지원 llama-cpp-python 설치 (선택 사항)
  CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python
  
  # AWQ 및 GPTQ 지원을 위한 추가 패키지 설치 (선택 사항)
  pip install -U autoawq auto-gptq optimum packaging
  ```

  ## 사용 방법

  ### 기본 사용법

  ```bash
  # 전체 파이프라인 실행 (기본 llama.cpp 방식)
  python -m llm_quantizer.main --model "kakaocorp/kanana-nano-2.1b-instruct" --output-dir "quantized_kanana"
  
  # 캘리브레이션 데이터 생성 포함
  python -m llm_quantizer.main --model "kakaocorp/kanana-nano-2.1b-instruct" --generate-calibration
  
  # AWQ 방식으로 양자화
  python -m llm_quantizer.main --model "kakaocorp/kanana-nano-2.1b-instruct" --method awq --output-dir "awq_kanana"
  
  # GPTQ 방식으로 양자화
  python -m llm_quantizer.main --model "kakaocorp/kanana-nano-2.1b-instruct" --method gptq --output-dir "gptq_kanana"
  
  # 양자화 방식 목록 확인
  python -m llm_quantizer.main --list-methods
  ```

  ### 고급 사용법

  ```bash
  # 특정 단계 스킵
  python -m llm_quantizer.main --model "kakaocorp/kanana-nano-2.1b-instruct" --skip-architecture --skip-conceptual
  
  # 설정 파일 사용
  python -m llm_quantizer.main --config configs/quantize_config.json
  
  # 설정 내보내기
  python -m llm_quantizer.main --model "kakaocorp/kanana-nano-2.1b-instruct" --export-config "configs/my_config.json"
  
  # 방식별 특수 설정 적용
  python -m llm_quantizer.main --model "kakaocorp/kanana-nano-2.1b-instruct" --method gptq --gptq-group-size 64 --gptq-act-order
  ```

  ## 모듈 구조

  - core/

    : 핵심 양자화 기능

    - **model_loader.py**: 모델 로딩 관련

    - **architecture_analyzer.py**: 모델 구조 분석

    - **importance_analyzer.py**: 레이어 중요도 분석

    - **quantization_planner.py**: 양자화 전략 설계

    - **quantizer.py**: 실제 양자화 구현

    - **dynamic_quantizer.py**: 통합 파이프라인

    - quantization/

      : 양자화 방식 모듈

      - **base.py**: 양자화 추상 기본 클래스
      - **factory.py**: 양자화 팩토리
      - **llamacpp.py**: llama.cpp 양자화 구현
      - **awq.py**: AWQ 양자화 구현
      - **gptq.py**: GPTQ 양자화 구현

  - calibration/

    : 캘리브레이션 관련

    - **data_generator.py**: 캘리브레이션 데이터 생성

  - utils/

    : 유틸리티 기능

    - **visualization.py**: 시각화 도구
    - **script_generator.py**: 스크립트 생성 도구
    - **config.py**: 설정 관련 기능

  - inference/

    : 추론 관련

    - **llama_cpp_python.py**: llama-cpp-python 래퍼
    - **validator.py**: 양자화 모델 검증

  ## 명령행 옵션

  - `--model`: 양자화할 모델 이름 (Hugging Face ID)
  - `--output-dir`: 출력 디렉토리
  - `--method`: 양자화 방식 (llamacpp, awq, gptq)
  - `--device`: 사용할 디바이스 (cuda 또는 cpu)
  - `--generate-calibration`: 모델 특화 캘리브레이션 데이터 생성
  - `--cal-samples`: 생성할 캘리브레이션 샘플 수
  - `--config`: 설정 파일 경로
  - `--export-config`: 설정을 파일로 내보내기
  - `--list-methods`: 사용 가능한 양자화 방식 목록 표시
  - `--skip-architecture`: 아키텍처 분석 스킵
  - `--skip-importance`: 중요도 분석 스킵
  - `--skip-conceptual`: 개념적 양자화 구현 스킵
  - `--verbose`: 상세 로깅 출력

  ### 방식별 특수 옵션

  #### llama.cpp 옵션

  - `--llamacpp-type`: 양자화 타입 (q4_0, q4_1, q5_0, q5_1, q8_0 등)

  #### AWQ 옵션

  - `--awq-zero-point`: AWQ 영점 조정 사용
  - `--awq-group-size`: AWQ 그룹 크기

  #### GPTQ 옵션

  - `--gptq-group-size`: GPTQ 그룹 크기
  - `--gptq-act-order`: GPTQ 활성화 순서 사용

  ## 양자화 결과물

  실행 후 다음과 같은 결과물이 생성됩니다:

  1. 아키텍처 분석 결과 (`architecture_analysis/`)
     - `architecture_analysis.json`: 모델 구조 분석 정보
     - `layer_distribution.png`: 레이어 분포 시각화
  2. 중요도 분석 결과 (`importance_analysis/`)
     - `importance_analysis.json`: 레이어별 중요도 정보
     - `weight_distributions.png`: 가중치 분포 시각화
  3. 양자화 계획 (`quantization_plan/`)
     - `quantization_plan.json`: 레이어별 양자화 비트 할당 계획
     - `quantization_plan.png`: 비트 분포 시각화
     - `quantization_command.sh`: 양자화 명령어 스크립트 (선택한 방식에 따라 달라짐)
  4. 캘리브레이션 데이터 (`calibration_data/`)
     - `calibration_data.txt`: llama.cpp용 캘리브레이션 데이터
     - `awq_calibration_data.jsonl`: AWQ용 캘리브레이션 데이터 (AWQ 선택 시)
     - `gptq_calibration_data.jsonl`: GPTQ용 캘리브레이션 데이터 (GPTQ 선택 시)
     - `calibration_pairs.json`: 프롬프트-응답 쌍 (분석용)
     - `use_calibration_data.sh`: 캘리브레이션 데이터 사용 스크립트
  5. 양자화 모델 (`quantized_model/`)
     - `conceptual_quantized_model.pt`: 개념적 양자화 결과 (Torch 모델)
     - 방식별 최종 양자화 모델 (스크립트 실행 후)

  ## 양자화 방식별 특징

  ### llama.cpp (GGUF)

  - **장점**: 다양한 비트 수 지원 (2~8비트), 높은 호환성
  - **특징**: 레이어별 다른 양자화 적용 가능, 캘리브레이션 지원
  - **용도**: 메모리 사용량 최적화 필요 시, CPU 추론 환경

  ### AWQ (Activation-aware Weight Quantization)

  - **장점**: 높은 정확도 유지, 빠른 추론 속도
  - **특징**: 활성화 인식 양자화, 주로 4비트 양자화
  - **용도**: GPU 추론 속도가 중요한 환경

  ### GPTQ (Generative Pre-trained Transformer Quantization)

  - **장점**: 높은 압축률, 적은 성능 손실
  - **특징**: 학습 후 양자화, 주로 3~4비트 양자화
  - **용도**: GPU 환경에서 메모리와 정확도 사이 균형 필요 시

  ## 양자화 실행 방법

  ### llama.cpp 양자화 실행

  ```bash
  # 양자화 명령어 스크립트 실행
  cd output_dir
  chmod +x quantization_plan/quantization_command.sh
  ./quantization_plan/quantization_command.sh
  
  # 반복 방지 설정으로 모델 실행
  cd output_dir
  chmod +x quantization_plan/prevent_repetition.sh
  ./quantization_plan/prevent_repetition.sh
  ```

  ### AWQ 양자화 실행

  ```bash
  # AWQ 양자화 스크립트 실행
  cd output_dir
  chmod +x scripts/awq_quantization.sh
  ./scripts/awq_quantization.sh
  ```

  ### GPTQ 양자화 실행

  ```bash
  # GPTQ 양자화 스크립트 실행
  cd output_dir
  chmod +x scripts/gptq_quantization.sh
  ./scripts/gptq_quantization.sh
  ```

  ## 새로운 양자화 방식 추가 방법

  새로운 양자화 방식을 추가하려면:

  1. `core/quantization/base.py`에 정의된 `QuantizationBase` 클래스를 상속하는 새 클래스를 생성합니다.
  2. 필요한 모든 메서드를 구현합니다.
  3. `core/quantization/factory.py`에 새 양자화 방식을 등록합니다.
  4. `quantize.py`에 새 방식에 대한 CLI 인자를 추가합니다.

  예:

  ```python
  # core/quantization/new_method.py
  from .base import QuantizationBase
  
  class NewMethodQuantization(QuantizationBase):
      # 필요한 메서드 구현
      ...
  ```

  ## 라이선스

  이 프로젝트는 GPL 3.0 라이선스로 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.

  ## 감사의 글

  이 프로젝트는 다음 오픈 소스 프로젝트에 기반합니다:

  - llama.cpp
  - transformers
  - llama-cpp-python
  - AutoAWQ
  - AutoGPTQ