# LLM 양자화 자동화 도구

LLM(Large Language Model) 양자화를 자동화하는 도구입니다. 이 도구는 모델의 아키텍처를 분석하고, 레이어별 중요도를 평가하여 최적의 양자화 전략을 설계하고 구현합니다.

## 주요 기능

1. **모델 아키텍처 분석**: 모델 구조와 레이어별 파라미터 수 분석
2. **슈퍼 웨이트 탐지**: 중요한 가중치 식별 및 보존
3. **동적 양자화 전략**: 레이어별 특성에 따른 최적의 비트 할당
4. **캘리브레이션 데이터 생성**: 모델 특화 캘리브레이션 데이터 자동 생성
5. **양자화 스크립트 생성**: llama.cpp 기반 양자화 자동화 스크립트
6. **반복 출력 방지**: 양자화 모델의 반복 출력 방지 설정

## 설치 방법

```bash
# 저장소 클론
git clone https://github.com/yourusername/llm-quantizer.git
cd llm-quantizer

# 의존성 설치
pip install -r requirements.txt

# CUDA 지원 llama-cpp-python 설치 (선택 사항)
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python
```

## 사용 방법

### 기본 사용법

```bash
# 전체 파이프라인 실행
python -m llm_quantizer.main --model "kakaocorp/kanana-nano-2.1b-instruct" --output-dir "quantized_kanana"

# 캘리브레이션 데이터 생성 포함
python -m llm_quantizer.main --model "kakaocorp/kanana-nano-2.1b-instruct" --generate-calibration
```

### 고급 사용법

```bash
# 특정 단계 스킵
python -m llm_quantizer.main --model "kakaocorp/kanana-nano-2.1b-instruct" --skip-architecture --skip-conceptual

# 설정 파일 사용
python -m llm_quantizer.main --config configs/quantize_config.json

# 설정 내보내기
python -m llm_quantizer.main --model "kakaocorp/kanana-nano-2.1b-instruct" --export-config "configs/my_config.json"
```

## 모듈 구조

- **core/**: 핵심 양자화 기능
  - **model_loader.py**: 모델 로딩 관련
  - **architecture_analyzer.py**: 모델 구조 분석
  - **importance_analyzer.py**: 레이어 중요도 분석
  - **quantization_planner.py**: 양자화 전략 설계
  - **quantizer.py**: 실제 양자화 구현
  - **dynamic_quantizer.py**: 통합 파이프라인

- **calibration/**: 캘리브레이션 관련
  - **data_generator.py**: 캘리브레이션 데이터 생성

- **utils/**: 유틸리티 기능
  - **visualization.py**: 시각화 도구
  - **script_generator.py**: 스크립트 생성 도구
  - **config.py**: 설정 관련 기능

- **inference/**: 추론 관련
  - **llama_cpp_python.py**: llama-cpp-python 래퍼
  - **validator.py**: 양자화 모델 검증

## 명령행 옵션

- `--model`: 양자화할 모델 이름 (Hugging Face ID)
- `--output-dir`: 출력 디렉토리
- `--device`: 사용할 디바이스 (cuda 또는 cpu)
- `--generate-calibration`: 모델 특화 캘리브레이션 데이터 생성
- `--cal-samples`: 생성할 캘리브레이션 샘플 수
- `--config`: 설정 파일 경로
- `--export-config`: 설정을 파일로 내보내기
- `--skip-architecture`: 아키텍처 분석 스킵
- `--skip-importance`: 중요도 분석 스킵
- `--skip-conceptual`: 개념적 양자화 구현 스킵

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
   - `quantization_command.sh`: llama.cpp 양자화 명령어 스크립트

4. 캘리브레이션 데이터 (`calibration_data/`)
   - `calibration_data.txt`: 생성된 캘리브레이션 데이터
   - `calibration_pairs.json`: 프롬프트-응답 쌍 (분석용)
   - `use_calibration_data.sh`: 캘리브레이션 데이터 사용 스크립트

5. 양자화 모델 (`quantized_model/`)
   - `conceptual_quantized_model.pt`: 개념적 양자화 결과 (Torch 모델)

## llama.cpp 양자화 실행 방법

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

## 기여 방법

1. 저장소 포크
2. 기능 브랜치 생성 (`git checkout -b feature/amazing-feature`)
3. 변경 사항 커밋 (`git commit -m 'Add some amazing feature'`)
4. 브랜치에 푸시 (`git push origin feature/amazing-feature`)
5. Pull Request 생성

## 라이선스

이 프로젝트는 GPL 3.0 라이선스로 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.

## 감사의 글

이 프로젝트는 다음 오픈 소스 프로젝트에 기반합니다:

- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [transformers](https://github.com/huggingface/transformers)
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)