#!/usr/bin/env python3
# main.py - 메인 실행 스크립트

import argparse
import os
import time
from typing import List, Dict, Any, Optional

from core.dynamic_quantizer import DynamicQuantizer
from calibration.data_generator import CalibrationDataGenerator
from utils.config import parse_config, save_config

def parse_arguments():
    """커맨드 라인 인자 파싱"""
    parser = argparse.ArgumentParser(description="LLM 동적 양자화 도구")
    
    parser.add_argument("--model", type=str, required=True,
                        help="양자화할 모델 이름 (Hugging Face ID)")
    parser.add_argument("--output-dir", type=str, default="quantized_model",
                        help="출력 디렉토리 (기본값: quantized_model)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="사용할 디바이스 (기본값: cuda)")
    parser.add_argument("--generate-calibration", action="store_true",
                        help="모델 특화 캘리브레이션 데이터 생성")
    parser.add_argument("--cal-samples", type=int, default=20,
                        help="생성할 캘리브레이션 샘플 수 (기본값: 20)")
    parser.add_argument("--config", type=str, 
                        help="설정 파일 경로")
    parser.add_argument("--export-config", type=str, 
                        help="설정을 파일로 내보내기")
    parser.add_argument("--skip-architecture", action="store_true",
                        help="아키텍처 분석 스킵")
    parser.add_argument("--skip-importance", action="store_true",
                        help="중요도 분석 스킵")
    parser.add_argument("--skip-conceptual", action="store_true",
                        help="개념적 양자화 구현 스킵")
    
    return parser.parse_args()

def main():
    """메인 실행 함수"""
    # 인자 파싱
    args = parse_arguments()
    
    # 설정 파일 사용 시
    if args.config:
        config = parse_config(args.config)
    else:
        config = vars(args)
    
    # 설정 내보내기
    if args.export_config:
        save_config(config, args.export_config)
        print(f"설정이 {args.export_config}에 저장되었습니다")
    
    start_time = time.time()
    print(f"LLM 양자화 자동화 시작: {args.model}")
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 캘리브레이션 데이터 생성 (요청된 경우)
    if args.generate_calibration:
        print(f"\n== 모델 특화 캘리브레이션 데이터 생성 ==")
        calibration_dir = os.path.join(args.output_dir, "calibration_data")
        generator = CalibrationDataGenerator(args.model, calibration_dir, args.device)
        cal_results = generator.run_generation_pipeline(args.cal_samples)
        print(f"캘리브레이션 데이터 생성 완료: {cal_results['calibration_path']}")
    
    # 양자화 파이프라인 초기화 및 실행
    print(f"\n== 동적 양자화 파이프라인 시작 ==")
    
    quantizer = DynamicQuantizer(args.model, args.output_dir, args.device)
    
    # 모델 로드
    quantizer.load_model()
    
    # 아키텍처 분석 (옵션에 따라 실행)
    if not args.skip_architecture:
        quantizer.analyze_architecture()
    
    # 중요도 분석 (옵션에 따라 실행)
    if not args.skip_importance:
        # 샘플 텍스트 (추후 확장 가능)
        sample_texts = [
            "인공지능의 미래에 대해 설명해주세요.",
            "파이썬으로 간단한 웹 크롤러를 만들어주세요.",
            "지속 가능한 발전이란 무엇이며 왜 중요한가요?"
        ]
        quantizer.analyze_layer_importance(sample_texts)
    
    # 양자화 전략 설계
    quantizer.design_quantization_strategy()
    
    # 개념적 양자화 구현 (옵션에 따라 실행)
    if not args.skip_conceptual:
        quantizer.implement_quantization()
    
    elapsed_time = time.time() - start_time
    print(f"\n== 양자화 파이프라인 완료 ==")
    print(f"총 소요 시간: {elapsed_time:.2f}초 ({elapsed_time/60:.2f}분)")
    print(f"결과 파일: {args.output_dir}")

if __name__ == "__main__":
    main()