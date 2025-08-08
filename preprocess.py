import pandas as pd
from scipy.sparse import coo_matrix
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import os

# --- 설정 (사용하는 데이터셋에 맞게 수정) ---
# MovieLens 100k 데이터셋을 사용할 경우
DATA_DIR = './Datasets/ml-100k/'
RAW_DATA_FILE = 'u.data'
SEPARATOR = '\t'
# -----------------------------------------

# --- 저장할 폴더 생성 ---
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
    print(f"'{DATA_DIR}' 폴더를 생성했습니다. 이 폴더에 원본 데이터 파일을 넣어주세요.")
# ---------------------

# 원본 데이터 파일 경로
raw_file_path = os.path.join(DATA_DIR, RAW_DATA_FILE)

print(f"'{raw_file_path}' 파일을 읽어 전처리를 시작합니다...")

try:
    # 1. 원본 데이터 로드 (u.data는 user_id, item_id, rating, timestamp 순서)
    df = pd.read_csv(
        raw_file_path,
        sep=SEPARATOR,
        header=None,
        names=['user_id', 'item_id', 'rating', 'timestamp'],
        engine='python'
    )

    # 2. 사용자 및 아이템 ID를 0부터 시작하도록 조정 (모델이 0-based index를 사용하기 때문)
    df['user_id'] = df['user_id'] - 1
    df['item_id'] = df['item_id'] - 1

    # 3. 전체 상호작용 데이터를 훈련(train) / 테스트(test) 세트로 분리 (80:20 비율)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    print("데이터를 훈련 세트와 테스트 세트로 분리했습니다.")

    # 4. 희소 행렬(Sparse Matrix) 생성
    # 모델이 사용할 수 있도록 상호작용 데이터를 행렬 형태로 변환
    num_users = df['user_id'].max() + 1
    num_items = df['item_id'].max() + 1

    # 훈련 세트 희소 행렬
    train_mat = coo_matrix(
        (np.ones(train_df.shape[0]), (train_df['user_id'], train_df['item_id'])),
        shape=(num_users, num_items)
    )

    # 테스트 세트 희소 행렬
    test_mat = coo_matrix(
        (np.ones(test_df.shape[0]), (test_df['user_id'], test_df['item_id'])),
        shape=(num_users, num_items)
    )

    print("훈련 및 테스트용 희소 행렬을 생성했습니다.")

    # 5. 생성된 행렬을 .pkl 파일로 저장
    with open(os.path.join(DATA_DIR, 'trnMat.pkl'), 'wb') as f:
        pickle.dump(train_mat, f)

    with open(os.path.join(DATA_DIR, 'tstMat.pkl'), 'wb') as f:
        pickle.dump(test_mat, f)

    print("="*40)
    print("🎉 성공! 필요한 .pkl 파일들이 생성되었습니다.")
    print(f"  - 훈련 데이터: {os.path.join(DATA_DIR, 'trnMat.pkl')}")
    print(f"  - 테스트 데이터: {os.path.join(DATA_DIR, 'tstMat.pkl')}")
    print("이제 Main.py를 실행하세요!")
    print("="*40)

except FileNotFoundError:
    print("-" * 40)
    print(f"🚨 오류: 원본 데이터 파일이 없습니다!")
    print(f"'{raw_file_path}' 경로에 '{RAW_DATA_FILE}' 파일이 있는지 확인해주세요.")
    print("-" * 40)

except Exception as e:
    print(f"예상치 못한 오류가 발생했습니다: {e}")