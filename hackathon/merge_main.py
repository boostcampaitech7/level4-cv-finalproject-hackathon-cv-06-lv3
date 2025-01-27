import argparse
import logging
import os
from typing import List

from src.preprocess.merge_files import merge_files_in_chunks  # 수정된 merge_files 함수 임포트

def get_files_in_directory(directory: str, extensions: List[str]) -> List[str]:
    """
    지정된 디렉토리에서 특정 확장자를 가진 모든 파일을 찾습니다.
    """
    files = []
    for file in os.listdir(directory):
        if any(file.endswith(ext) for ext in extensions):
            files.append(os.path.join(directory, file))
    return sorted(files)  # 파일을 정렬하여 순서 보장

def main(args):
    # 로깅 설정
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # 입력 파일 경로 리스트
    input_dir = args.input_dir
    supported_extensions = ['.csv', '.parquet']  # 지원하는 파일 형식
    input_files = get_files_in_directory(input_dir, supported_extensions)

    if len(input_files) != 5:
        logging.error(f"{input_dir} 디렉토리에서 정확히 5개의 파일이 필요합니다. 현재 파일 개수: {len(input_files)}")
        return

    logging.info(f"병합할 파일 목록: {input_files}")

    # 출력 파일 경로
    output_file = args.output
    os.makedirs(os.path.dirname(output_file), exist_ok=True)  # 디렉토리가 없으면 생성

    # 파일 병합 실행
    try:
        merge_files_in_chunks(input_files, output_file, chunk_size=args.chunk_size)
        logging.info(f"파일 병합이 완료되었습니다. 결과는 {output_file}에 저장되었습니다.")
    except Exception as e:
        logging.error(f"파일 병합 중 오류 발생: {e}")

if __name__ == "__main__":
    # argparse 설정
    parser = argparse.ArgumentParser(description='5개의 파일을 공통 컬럼 기준으로 병합합니다.')
    parser.add_argument(
        '--input_dir',
        '-i',
        type=str,
        default='/data/ephemeral/home/level4-cv-finalproject-hackathon-cv-06-lv3/hackathon/data/raw',
        help='병합할 파일이 있는 디렉토리 경로를 입력하세요'
    )
    parser.add_argument(
        '--output',
        '-o',
        type=str,
        default='/data/ephemeral/home/level4-cv-finalproject-hackathon-cv-06-lv3/hackathon/data/raw/merged_data.parquet',
        help='병합 결과를 저장할 파일 경로를 입력하세요'
    )
    parser.add_argument(
        '--chunk_size',
        '-c',
        type=int,
        default=20000000,
        help='log_data를 청크 단위로 처리할 크기를 입력하세요 (기본값: 100,000)'
    )
    args = parser.parse_args()

    # main 함수 실행
    main(args)
