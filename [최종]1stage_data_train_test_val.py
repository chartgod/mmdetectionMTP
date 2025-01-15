import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image

def create_directories(base_dir, subdirs):
    """디렉토리를 생성하는 헬퍼 함수."""
    for subdir in subdirs:
        os.makedirs(os.path.join(base_dir, subdir), exist_ok=True)

def match_files(base_dir, output_dir):
    """A와 B 디렉토리의 파일을 매칭하고 CSV로 저장."""
    print("1단계: 파일 매칭 및 CSV 생성 중...")

    image_dir = os.path.join(base_dir, 'image')
    label_dir = os.path.join(base_dir, 'label')

    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"이미지 디렉토리가 존재하지 않습니다: {image_dir}")
    if not os.path.exists(label_dir):
        raise FileNotFoundError(f"레이블 디렉토리가 존재하지 않습니다: {label_dir}")

    a_dir = os.path.join(output_dir, 'A')
    b_dir = os.path.join(output_dir, 'B')
    output_label_dir = os.path.join(output_dir, 'label')

    create_directories(output_dir, ['A', 'B', 'label'])

    records = []
    count = 1

    for sub_dir in os.listdir(image_dir):
        image_sub_dir = os.path.join(image_dir, sub_dir)
        if not os.path.isdir(image_sub_dir):
            continue

        b_dirs = [d for d in os.listdir(image_sub_dir) if d.endswith('_B')]
        a_dirs = [d for d in os.listdir(image_sub_dir) if d.endswith('_A')]

        for b_dir_name, a_dir_name in zip(b_dirs, a_dirs):
            b_dir_path = os.path.join(image_sub_dir, b_dir_name)
            a_dir_path = os.path.join(image_sub_dir, a_dir_name)

            b_files = sorted([f for f in os.listdir(b_dir_path) if f.endswith('.png')])
            a_files = sorted([f for f in os.listdir(a_dir_path) if f.endswith('.png')])

            for b_file, a_file in zip(b_files, a_files):
                new_file_name = f"{count}.png"

                b_file_path = os.path.join(b_dir_path, b_file)
                shutil.copyfile(b_file_path, os.path.join(b_dir, new_file_name))

                a_file_path = os.path.join(a_dir_path, a_file)
                shutil.copyfile(a_file_path, os.path.join(a_dir, new_file_name))

                records.append({
                    'A': a_file_path,
                    'B': b_file_path,
                    'Label': None,
                    'count': count
                })

                count += 1

    label_count = 1
    for root, _, files in os.walk(label_dir):
        for file in sorted(files):
            if file.endswith('.png'):
                original_label_path = os.path.join(root, file)
                new_label_file = f"{label_count}.png"
                shutil.copyfile(original_label_path, os.path.join(output_label_dir, new_label_file))

                if label_count <= len(records):
                    records[label_count - 1]['Label'] = original_label_path

                label_count += 1

    csv_path = os.path.join(output_dir, 'file_records.csv')
    pd.DataFrame(records).to_csv(csv_path, index=False, encoding='cp949')
    print(f"CSV 파일이 저장되었습니다: {csv_path}")

    return output_dir

def split_files(output_dir):
    """파일을 학습, 테스트, 검증 세트로 분할."""
    print("2단계: 파일을 학습(train), 테스트(test), 검증(val) 세트로 분할 중...")

    dirs = ['A', 'B', 'label']
    output_dirs = ['train', 'test', 'val']

    for output_dir_name in output_dirs:
        create_directories(output_dir, [os.path.join(output_dir_name, d) for d in dirs])

    for d in dirs:
        dir_path = os.path.join(output_dir, d)
        files = [f for f in os.listdir(dir_path) if f.endswith('.png')]

        train_files, temp_files = train_test_split(files, test_size=0.2, random_state=42)
        test_files, val_files = train_test_split(temp_files, test_size=0.5, random_state=42)

        for file_list, output_dir_name in zip([train_files, test_files, val_files], output_dirs):
            for file_name in file_list:
                src = os.path.join(dir_path, file_name)
                dst = os.path.join(output_dir, output_dir_name, d, file_name)
                shutil.copyfile(src, dst)

    print("파일 분할이 완료되었습니다.")

def crop_images(root_dir, crop_size=256):
    """이미지를 작은 패치로 크롭."""
    print("3단계: 이미지를 작은 패치로 크롭 중...")

    output_dir = os.path.join(root_dir, 'output')  # 모든 데이터를 output 디렉터리로 저장
    create_directories(output_dir, ['train', 'test', 'val'])

    subsets = ['train', 'test', 'val']
    subdirs = ['A', 'B', 'label']

    for subset in subsets:
        print(f"  - {subset} 데이터셋 크롭 중...")
        input_subset_dir = os.path.join(root_dir, subset)
        output_subset_dir = os.path.join(output_dir, subset)

        for subdir in subdirs:
            input_subdir = os.path.join(input_subset_dir, subdir)
            output_subdir = os.path.join(output_subset_dir, subdir)

            os.makedirs(output_subdir, exist_ok=True)

            files = [f for f in os.listdir(input_subdir) if f.endswith(".png")]
            total_files = len(files)

            for idx, file in enumerate(files, start=1):
                image_path = os.path.join(input_subdir, file)
                crop_image(image_path, output_subdir, crop_size)
                if idx % 10 == 0 or idx == total_files:
                    print(f"    {subdir}: {idx}/{total_files} 파일 크롭 완료")

    print("이미지 크롭이 완료되었습니다.")

def crop_image(image_path, output_dir, crop_size=256):
    """단일 이미지를 크롭하는 헬퍼 함수."""
    img = Image.open(image_path)
    img_width, img_height = img.size

    for i in range(0, img_width, crop_size):
        for j in range(0, img_height, crop_size):
            box = (i, j, min(i + crop_size, img_width), min(j + crop_size, img_height))
            cropped_img = img.crop(box)
            cropped_img.save(os.path.join(output_dir, f"{os.path.basename(image_path).split('.')[0]}_{i}_{j}.png"))
def main():
    import sys

    # 실행 파일 또는 스크립트의 디렉터리 경로
    if getattr(sys, 'frozen', False):
        # PyInstaller로 빌드된 실행 파일의 디렉터리
        exe_dir = os.path.dirname(sys.executable)
    else:
        # 소스 코드 실행 시의 디렉터리
        exe_dir = os.path.dirname(os.path.abspath(__file__))

    # 데이터 디렉터리와 출력 디렉터리 설정
    input_dir = os.path.abspath(os.path.join(exe_dir, '..', '3_data_transform', 'output_learning_data'))
    output_dir = os.path.join(exe_dir, 'input_train_data')

    # 경로 출력 (디버깅용)
    print(f"실행 파일 디렉터리: {exe_dir}")
    print(f"입력 디렉터리: {input_dir}")
    print(f"출력 디렉터리: {output_dir}")

    # 입력 데이터 디렉터리 존재 여부 확인
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"입력 디렉터리를 찾을 수 없습니다: {input_dir}")

    # 출력 디렉터리 생성
    create_directories(exe_dir, ['input_train_data'])

    # 작업 실행
    match_files(input_dir, output_dir)
    split_files(output_dir)
    crop_images(output_dir)

if __name__ == "__main__":
    main()
