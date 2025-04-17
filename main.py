#!/usr/bin/env python3
# main.py - LLM 양자화 자동화 도구 메인 스크립트

import os
import sys
import argparse
import logging
import json
from typing import Dict, Any

# 로깅 설정
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 모듈 임포트
try:
    from core.model_loader import load_model_and_tokenizer
    from core.architecture_analyzer import analyze_architecture
    from core.importance_analyzer import analyze_importance
    from core.quantization_planner import create_quantization_plan
    from core.dynamic_quantizer import quantize_model_conceptually
    from core.quantization import QuantizationFactory
    from calibration.data_generator import generate_calibration_data
    from utils.visualization import visualize_architecture, visualize_importance, visualize_quantization_plan
    from utils.script_generator import generate_quantization_script, generate_repetition_prevention_script
    from utils.config import load_config, save_config
except ImportError as e:
    logger.error(f"모듈 로드 중 오류: {e}")
    logger.error("필요한 모든 의존성이 설치되어 있는지 확인하세요.")
    sys.exit(1)

def setup_parser() -> argparse.ArgumentParser:
    """
    명령행 인자 파서 설정
    
    Returns:
        argparse.ArgumentParser: 설정된 인자 파서
    """
    parser = argparse.ArgumentParser(
        description="LLM 양자화 자동화 도구",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 기본 인자
    parser.add_argument(
        "--model", 
        type=str, 
        required=True, 
        help="양자화할 모델 경로 (Hugging Face 모델명 또는 로컬 경로)"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="./output", 
        help="출력 디렉토리"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda", 
        help="사용할 디바이스 (cuda 또는 cpu)"
    )
    
    # 캘리브레이션 관련 인자
    parser.add_argument(
        "--generate-calibration", 
        action="store_true", 
        help="모델 특화 캘리브레이션 데이터 생성"
    )
    parser.add_argument(
        "--cal-samples", 
        type=int, 
        default=32, 
        help="생성할 캘리브레이션 샘플 수"
    )
    
    # 설정 관련 인자
    parser.add_argument(
        "--config", 
        type=str, 
        default=None, 
        help="설정 파일 경로"
    )
    parser.add_argument(
        "--export-config", 
        type=str, 
        default=None, 
        help="설정을 파일로 내보내기"
    )
    
    # 양자화 방식 관련 인자
    parser.add_argument(
        "--method", 
        type=str, 
        choices=['llamacpp', 'awq', 'gptq'],
        default="llamacpp", 
        help="양자화 방식 선택"
    )
    
    parser.add_argument(
        "--list-methods", 
        action="store_true", 
        help="사용 가능한 양자화 방식 목록 표시"
    )
    
    # 파이프라인 단계 제어
    parser.add_argument(
        "--skip-architecture", 
        action="store_true", 
        help="아키텍처 분석 스킵"
    )
    parser.add_argument(
        "--skip-importance", 
        action="store_true", 
        help="중요도 분석 스킵"
    )
    parser.add_argument(
        "--skip-conceptual", 
        action="store_true", 
        help="개념적 양자화 구현 스킵"
    )
    
    # 방식별 특수 인자
    # GPTQ 관련 인자
    gptq_group = parser.add_argument_group("GPTQ 특수 설정")
    gptq_group.add_argument(
        "--gptq-group-size", 
        type=int, 
        default=128, 
        help="GPTQ 그룹 크기"
    )
    gptq_group.add_argument(
        "--gptq-act-order", 
        action="store_true", 
        help="GPTQ 활성화 순서 사용"
    )
    
    # AWQ 관련 인자
    awq_group = parser.add_argument_group("AWQ 특수 설정")
    awq_group.add_argument(
        "--awq-zero-point", 
        action="store_true", 
        help="AWQ 영점 조정 사용"
    )
    awq_group.add_argument(
        "--awq-group-size", 
        type=int, 
        default=128, 
        help="AWQ 그룹 크기"
    )
    
    # llama.cpp 관련 인자
    llamacpp_group = parser.add_argument_group("llama.cpp 특수 설정")
    llamacpp_group.add_argument(
        "--llamacpp-type", 
        type=str, 
        default="q4_k", 
        help="llama.cpp 양자화 타입 (q4_0, q4_1, q5_0, q5_1, q8_0 등)"
    )
    
    # 기타 인자
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="상세 로깅 출력"
    )
    
    return parser

def print_methods_info():
    """사용 가능한 양자화 방식 및 설명 출력"""
    try:
        methods = QuantizationFactory.get_available_methods()
        
        print("\n=== 사용 가능한 양자화 방식 ===\n")
        
        for method, desc in methods.items():
            print(f"- {method}: {desc}")
            print(f"  {QuantizationFactory.get_method_description(method)}")
            print()
    except Exception as e:
        logger.error(f"양자화 방식 정보 출력 중 오류: {e}")

def get_quantization_config_from_args(args) -> Dict[str, Any]:
    """명령행 인자에서 양자화 설정 추출"""
    config = {}
    
    # 방식별 특수 설정
    if args.method == "gptq":
        config["method"] = "gptq"
        config["group_size"] = args.gptq_group_size
        config["act_order"] = args.gptq_act_order
    
    elif args.method == "awq":
        config["method"] = "awq"
        config["q_group_size"] = args.awq_group_size
        config["zero_point"] = args.awq_zero_point
    
    elif args.method == "llamacpp":
        config["method"] = "llamacpp"
        config["quantization_type"] = args.llamacpp_type
    
    return config

def main():
    """메인 함수"""
    # 명령행 인자 파싱
    parser = setup_parser()
    args = parser.parse_args()
    
    # 로그 레벨 설정
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 방식 목록 출력
    if args.list_methods:
        print_methods_info()
        sys.exit(0)
    
    # 설정 로드
    config = {}
    if args.config:
        try:
            config = load_config(args.config)
            logger.info(f"설정 파일 로드됨: {args.config}")
        except Exception as e:
            logger.error(f"설정 파일 로드 중 오류: {e}")
            sys.exit(1)
    
    # 명령행 인자를 설정에 추가
    config["model"] = args.model
    config["output_dir"] = args.output_dir
    config["device"] = args.device
    config["generate_calibration"] = args.generate_calibration
    config["cal_samples"] = args.cal_samples
    config["method"] = args.method
    
    # 양자화 방식별 특수 설정 추가
    quantization_config = get_quantization_config_from_args(args)
    if "quantization_config" not in config:
        config["quantization_config"] = {}
    config["quantization_config"].update(quantization_config)
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 설정 파일 내보내기
    if args.export_config:
        try:
            save_config(config, args.export_config)
            logger.info(f"설정 파일 저장됨: {args.export_config}")
        except Exception as e:
            logger.error(f"설정 파일 저장 중 오류: {e}")
    
    # 디렉토리 구조 생성
    arch_dir = os.path.join(args.output_dir, "architecture_analysis")
    imp_dir = os.path.join(args.output_dir, "importance_analysis")
    quant_dir = os.path.join(args.output_dir, "quantization_plan")
    cal_dir = os.path.join(args.output_dir, "calibration_data")
    model_dir = os.path.join(args.output_dir, "quantized_model")
    script_dir = os.path.join(args.output_dir, "scripts")
    
    for directory in [arch_dir, imp_dir, quant_dir, cal_dir, model_dir, script_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # 모델 및 토크나이저 로드
    logger.info(f"모델 로드 중: {args.model}")
    try:
        model, tokenizer = load_model_and_tokenizer(args.model, args.device)
    except Exception as e:
        logger.error(f"모델 로드 중 오류: {e}")
        sys.exit(1)
    
    # 아키텍처 분석
    if not args.skip_architecture:
        logger.info("모델 아키텍처 분석 중...")
        try:
            arch_info = analyze_architecture(model)
            arch_path = os.path.join(arch_dir, "architecture_analysis.json")
            with open(arch_path, 'w') as f:
                json.dump(arch_info, f, indent=2)
            
            # 아키텍처 시각화
            visualize_architecture(arch_info, os.path.join(arch_dir, "layer_distribution.png"))
            logger.info(f"아키텍처 분석 완료: {arch_path}")
        except Exception as e:
            logger.error(f"아키텍처 분석 중 오류: {e}")
            arch_info = {}
    else:
        logger.info("아키텍처 분석 스킵됨")
        arch_info = {}
    
    # 중요도 분석
    if not args.skip_importance:
        logger.info("레이어 중요도 분석 중...")
        try:
            importance_info = analyze_importance(model)
            imp_path = os.path.join(imp_dir, "importance_analysis.json")
            with open(imp_path, 'w') as f:
                json.dump(importance_info, f, indent=2)
            
            # 중요도 시각화
            visualize_importance(importance_info, os.path.join(imp_dir, "weight_distributions.png"))
            logger.info(f"중요도 분석 완료: {imp_path}")
        except Exception as e:
            logger.error(f"중요도 분석 중 오류: {e}")
            importance_info = {}
    else:
        logger.info("중요도 분석 스킵됨")
        importance_info = {}
    
    # 양자화 계획 생성
    logger.info("양자화 계획 생성 중...")
    try:
        quantization_plan = create_quantization_plan(arch_info, importance_info)
        plan_path = os.path.join(quant_dir, "quantization_plan.json")
        with open(plan_path, 'w') as f:
            json.dump(quantization_plan, f, indent=2)
        
        # 양자화 계획 시각화
        visualize_quantization_plan(quantization_plan, os.path.join(quant_dir, "quantization_plan.png"))
        logger.info(f"양자화 계획 생성 완료: {plan_path}")
    except Exception as e:
        logger.error(f"양자화 계획 생성 중 오류: {e}")
        quantization_plan = {}
    
    # 캘리브레이션 데이터 생성
    if args.generate_calibration:
        logger.info("캘리브레이션 데이터 생성 중...")
        try:
            cal_data = generate_calibration_data(model, tokenizer, args.cal_samples)
            
            # 방식별로 적절한 형식으로 캘리브레이션 데이터 저장
            if args.method == "llamacpp":
                cal_path = os.path.join(cal_dir, "calibration_data.txt")
                with open(cal_path, 'w') as f:
                    f.write(cal_data["text"])
            elif args.method == "awq":
                cal_path = os.path.join(cal_dir, "awq_calibration_data.jsonl")
                with open(cal_path, 'w') as f:
                    for item in cal_data["items"]:
                        f.write(json.dumps({"text": item}, ensure_ascii=False) + "\n")
            elif args.method == "gptq":
                cal_path = os.path.join(cal_dir, "gptq_calibration_data.jsonl")
                with open(cal_path, 'w') as f:
                    for i, item in enumerate(cal_data["items"]):
                        f.write(json.dumps({"text": item, "id": f"sample_{i}"}, ensure_ascii=False) + "\n")
            
            # 프롬프트-응답 쌍 저장 (분석용)
            pairs_path = os.path.join(cal_dir, "calibration_pairs.json")
            with open(pairs_path, 'w') as f:
                json.dump(cal_data["pairs"], f, indent=2)
            
            logger.info(f"캘리브레이션 데이터 생성 완료: {cal_path}")
        except Exception as e:
            logger.error(f"캘리브레이션 데이터 생성 중 오류: {e}")
    
    # 개념적 양자화 구현
    if not args.skip_conceptual:
        logger.info("개념적 양자화 구현 중...")
        try:
            conceptual_model = quantize_model_conceptually(model, quantization_plan)
            conceptual_path = os.path.join(model_dir, "conceptual_quantized_model.pt")
            torch.save(conceptual_model.state_dict(), conceptual_path)
            logger.info(f"개념적 양자화 구현 완료: {conceptual_path}")
        except Exception as e:
            logger.error(f"개념적 양자화 구현 중 오류: {e}")
    else:
        logger.info("개념적 양자화 구현 스킵됨")
    
    # 양자화 스크립트 생성
    logger.info(f"{args.method} 양자화 스크립트 생성 중...")
    try:
        script_path = generate_quantization_script(
            args.model, 
            quantization_plan, 
            method=args.method,
            output_dir=args.output_dir
        )
        logger.info(f"양자화 스크립트 생성 완료: {script_path}")
        
        # 반복 방지 스크립트 생성 (llama.cpp용)
        if args.method == "llamacpp":
            rep_script_path = generate_repetition_prevention_script(args.output_dir)
            logger.info(f"반복 방지 스크립트 생성 완료: {rep_script_path}")
    except Exception as e:
        logger.error(f"스크립트 생성 중 오류: {e}")
    
    logger.info(f"양자화 준비 완료! 다음 명령으로 양자화를 실행하세요:")
    
    if args.method == "llamacpp":
        print(f"\ncd {args.output_dir}")
        print(f"chmod +x quantization_plan/quantization_command.sh")
        print(f"./quantization_plan/quantization_command.sh")
    elif args.method == "awq":
        print(f"\ncd {args.output_dir}")
        print(f"chmod +x scripts/awq_quantization.sh")
        print(f"./scripts/awq_quantization.sh")
    elif args.method == "gptq":
        print(f"\ncd {args.output_dir}")
        print(f"chmod +x scripts/gptq_quantization.sh")
        print(f"./scripts/gptq_quantization.sh")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())