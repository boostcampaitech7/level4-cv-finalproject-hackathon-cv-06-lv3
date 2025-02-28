import sys
import fireducks.pandas as pd
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.dynamic_pipeline import preprocess_dynamic

def main():
    """
    실행 엔트리 포인트
    사용 예:
      python preprocess_main.py /path/to/data.csv
      python preprocess_main.py /path/to/data.parquet
    """
    if len(sys.argv) < 2:
        print("Usage: python preprocess_main.py <data_path>")
        sys.exit(1)

    data_path = sys.argv[1]
    file_extension = os.path.splitext(data_path)[-1].lower()

    # 데이터 로드
    try:
        if file_extension == ".csv":
            df = pd.read_csv(data_path)
        elif file_extension == ".parquet":
            df = pd.read_parquet(data_path)
        else:
            print(f"지원되지 않는 파일 형식: {file_extension}")
            sys.exit(1)
    except Exception as e:
        print(f"파일 로드 실패: {e}")
        sys.exit(1)

    print(f"데이터셋 로드 성공: {data_path} (shape={df.shape})")

    # 동적 전처리 실행
    try:
        processed_result = preprocess_dynamic(df)  # dict 반환
        
        # processed_df만 추출
        if isinstance(processed_result, dict):
            processed_df = processed_result.get("processed_df")
        else:
            print("Warning: preprocess_dynamic() returned a non-dictionary object. Converting to DataFrame...")
            processed_df = pd.DataFrame(processed_result)  

        # 전처리된 데이터가 정상적인지 확인
        if processed_df is None or not isinstance(processed_df, pd.DataFrame):
            print("Error: preprocess_dynamic() did not return a valid DataFrame.")
            sys.exit(1)

        print("전처리 완료!")
        print(f"전처리 후 shape: {processed_df.shape}")

    except Exception as e:
        print(f"전처리 중 오류 발생: {e}")
        sys.exit(1)

    # 전처리 결과 저장 (원본 확장자 유지)
    output_path = data_path.replace(file_extension, f"_processed{file_extension}")

    try:
        if file_extension == ".csv":
            processed_df.to_csv(output_path, index=False)
        elif file_extension == ".parquet":
            processed_df.to_parquet(output_path, index=False)
        
        print(f"전처리 결과 저장 완료: {output_path}")

    except Exception as e:
        print(f"결과 저장 중 오류 발생: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
