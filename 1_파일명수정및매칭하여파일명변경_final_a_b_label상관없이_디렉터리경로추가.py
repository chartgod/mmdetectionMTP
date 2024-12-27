import os
import pandas as pd

# 경로 설정
base_dir = '/home/lsh/share/CD/open-cd/data/5_test/learning'
image_dir = os.path.join(base_dir, 'image')
label_dir = os.path.join(base_dir, 'label1')

# A, B, 라벨 디렉터리 경로 설정
a_dir = os.path.join(base_dir, 'A')
b_dir = os.path.join(base_dir, 'B')
output_label_dir = os.path.join(base_dir, '라벨')

# 파일 리스트 및 매칭을 위한 변수 초기화
records = []
count = 1

# image 디렉터리 내 모든 상위 하위 디렉터리 탐색
for sub_dir in os.listdir(image_dir):
    image_sub_dir = os.path.join(image_dir, sub_dir)

    # 상위 하위 디렉터리가 존재하지 않으면 건너뜀
    if not os.path.isdir(image_sub_dir):
        continue

    # 상위 하위 디렉터리 내에서 *_B와 *_A로 끝나는 디렉터리 탐색
    b_dirs = [d for d in os.listdir(image_sub_dir) if d.endswith('_B')]
    a_dirs = [d for d in os.listdir(image_sub_dir) if d.endswith('_A')]

    # _B와 _A 디렉터리 매칭
    for b_dir_name, a_dir_name in zip(b_dirs, a_dirs):
        b_dir_path = os.path.join(image_sub_dir, b_dir_name)
        a_dir_path = os.path.join(image_sub_dir, a_dir_name)

        if not os.path.isdir(b_dir_path) or not os.path.isdir(a_dir_path):
            continue

        # _B 디렉터리의 파일 목록 정렬하여 순차적으로 매칭
        b_files = sorted([f for f in os.listdir(b_dir_path) if f.endswith('.png')])
        a_files = sorted([f for f in os.listdir(a_dir_path) if f.endswith('.png')])

        for b_file, a_file in zip(b_files, a_files):
            # 기록: 원본 파일명을 CSV에 저장
            records.append({
                'A': os.path.join(b_dir_path, a_file),
                'B': os.path.join(a_dir_path, b_file),
                'Label': None,  # 라벨은 나중에 추가
                'count': count
            })
            count += 1

# 라벨 디렉터리 내 모든 PNG 파일 순차적으로 경로 기록
label_count = 1
for root, _, files in os.walk(label_dir):
    for file in sorted(files):
        if file.endswith('.png'):
            original_label_file = os.path.join(root, file)  # 라벨 파일의 원본 이름

            # 라벨 파일명 기록 추가
            if label_count <= len(records):
                records[label_count - 1]['Label'] = original_label_file  # 원본 경로 저장

            label_count += 1

# CSV 파일 저장 (원본 파일명으로 기록)
df = pd.DataFrame(records)
csv_path = os.path.join(base_dir, 'file_record_dir.csv')
df.to_csv(csv_path, index=False, encoding='cp949')
print(f"CSV 파일이 {csv_path}에 저장되었습니다.")
