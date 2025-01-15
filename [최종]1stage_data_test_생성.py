import os
import shutil
import pandas as pd
from PIL import Image

def create_directories(base_dir, subdirs):
    """디렉토리를 생성하는 헬퍼 함수."""
    for subdir in subdirs:
        os.makedirs(os.path.join(base_dir, subdir), exist_ok=True)

def find_label_file(label_dir, filename):
    """
    label_dir 아래 모든 하위 폴더를 os.walk로 순회하며,
    특정 filename(문자열)이 있으면 해당 절대 경로를 반환.
    찾지 못하면 None.
    """
    for root, dirs, files in os.walk(label_dir):
        if filename in files:
            return os.path.join(root, filename)
    return None

def match_files(base_dir, output_dir):
    """
    A와 B 디렉토리의 파일을 매칭하고, CSV로 저장합니다.
    - B 디렉토리의 파일명을 기준으로 '_B' → '_Label' 로 대체하여 라벨 유무를 판별.
    - label 디렉터리에 해당 라벨이 있다면 복사하여 output_dir/label 폴더에 저장.
    - 라벨이 없는 경우(None)로 처리.
    - 결과적으로 output_dir에는 A, B, (선택) label 폴더가 생성되고,
      file_records.csv 파일이 생성됩니다.
    """
    print("1단계: 파일 매칭 및 CSV 생성 중...")

    image_dir = os.path.join(base_dir, 'image')
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"이미지 디렉토리가 존재하지 않습니다: {image_dir}")

    # label 디렉터리 존재 여부 확인
    label_dir = os.path.join(base_dir, 'label')
    has_label = os.path.exists(label_dir)

    # 출력 디렉터리에 필요한 디렉터리 생성
    if has_label:
        create_directories(output_dir, ['A', 'B', 'label'])
    else:
        create_directories(output_dir, ['A', 'B'])

    records = []
    count = 1

    # (1) A, B 디렉토리를 순회하며 파일 매칭
    for sub_dir in os.listdir(image_dir):
        image_sub_dir = os.path.join(image_dir, sub_dir)
        if not os.path.isdir(image_sub_dir):
            continue

        # _A, _B 로 끝나는 하위 폴더를 찾음
        b_dirs = [d for d in os.listdir(image_sub_dir) if d.endswith('_B')]
        a_dirs = [d for d in os.listdir(image_sub_dir) if d.endswith('_A')]

        # A/B 폴더 개수가 1:1이라고 가정하고 zip 사용
        for b_dir_name, a_dir_name in zip(b_dirs, a_dirs):
            b_dir_path = os.path.join(image_sub_dir, b_dir_name)
            a_dir_path = os.path.join(image_sub_dir, a_dir_name)

            b_files = sorted([f for f in os.listdir(b_dir_path) if f.endswith('.png')])
            a_files = sorted([f for f in os.listdir(a_dir_path) if f.endswith('.png')])

            # A/B 파일 개수도 동일하다고 가정 (zip 사용)
            for b_file, a_file in zip(b_files, a_files):
                new_file_name = f"{count}.png"

                # B 복사
                b_file_path = os.path.join(b_dir_path, b_file)
                out_b_path = os.path.join(output_dir, 'B', new_file_name)
                shutil.copyfile(b_file_path, out_b_path)

                # A 복사
                a_file_path = os.path.join(a_dir_path, a_file)
                out_a_path = os.path.join(output_dir, 'A', new_file_name)
                shutil.copyfile(a_file_path, out_a_path)

                # (2) label 파일 매칭 (has_label이면 수행)
                label_file_path = None
                if has_label:
                    # B 파일명 "XXX_B.png" → "XXX_Label.png" 로 추론
                    b_stem = b_file.rsplit('_B.png', 1)[0]  # 예: "SAMPLE_01_B.png" → "SAMPLE_01"
                    guessed_label_name = b_stem + '_Label.png'  # "SAMPLE_01_Label.png"

                    # label 디렉터리(하위 포함)에서 guessed_label_name을 찾기
                    possible_path = find_label_file(label_dir, guessed_label_name)
                    if possible_path is not None:
                        # 라벨 파일 복사
                        out_label_path = os.path.join(output_dir, 'label', new_file_name)
                        shutil.copyfile(possible_path, out_label_path)
                        label_file_path = possible_path

                # CSV 기록용
                record = {
                    'A': a_file_path,
                    'B': b_file_path,
                    'Label': label_file_path,  # None이면 라벨 없음
                    'count': count
                }
                records.append(record)
                count += 1

    # (3) CSV 파일 생성
    csv_path = os.path.join(output_dir, 'file_records.csv')
    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False, encoding='cp949')
    print(f"CSV 파일이 저장되었습니다: {csv_path}")

    return output_dir

def crop_image(image_path, output_dir, crop_size=256):
    """
    단일 이미지를 crop_size 단위로 잘라서 output_dir에 저장합니다.
    예: 256x256 크기로 여러 장 생성.
    """
    img = Image.open(image_path)
    img_width, img_height = img.size

    base_name = os.path.splitext(os.path.basename(image_path))[0]  # 파일명 (확장자 제외)
    for i in range(0, img_width, crop_size):
        for j in range(0, img_height, crop_size):
            box = (i, j, min(i + crop_size, img_width), min(j + crop_size, img_height))
            cropped_img = img.crop(box)
            # 파일명에 크롭 시작 좌표(i, j)를 포함
            new_file_name = f"{base_name}_{i}_{j}.png"
            cropped_img.save(os.path.join(output_dir, new_file_name))

def crop_images(root_dir, crop_size=256):
    """
    match_files()에서 만든 A, B, label 폴더의 이미지를
    crop_size 크기로 크롭하여 다시 A, B, label 폴더에 저장합니다.
    (크롭 전 원본 이미지는 *_original 폴더로 임시 이동 후, 작업 끝나면 삭제)
    라벨 폴더가 없으면 스킵합니다.
    """
    print("2단계: 이미지를 작은 패치로 크롭하여 최종 디렉터리에 저장 중...")

    # 라벨 폴더가 있는지 확인
    has_label = os.path.exists(os.path.join(root_dir, 'label'))

    # (1) 기존 폴더(A, B, label)를 *_original로 이름 변경하여 임시 이동
    a_original = os.path.join(root_dir, 'A_original')
    b_original = os.path.join(root_dir, 'B_original')
    os.rename(os.path.join(root_dir, 'A'), a_original)
    os.rename(os.path.join(root_dir, 'B'), b_original)

    original_folders = [a_original, b_original]
    label_original = None

    if has_label:
        label_original = os.path.join(root_dir, 'label_original')
        os.rename(os.path.join(root_dir, 'label'), label_original)
        original_folders.append(label_original)

    # (2) 새로 빈 폴더 생성
    if has_label:
        create_directories(root_dir, ['A', 'B', 'label'])
        subdirs = ['A', 'B', 'label']
        tmp_map = {
            'A': a_original,
            'B': b_original,
            'label': label_original
        }
    else:
        create_directories(root_dir, ['A', 'B'])
        subdirs = ['A', 'B']
        tmp_map = {
            'A': a_original,
            'B': b_original,
        }

    # (3) *_original 폴더 안의 이미지를 크롭하여 새 폴더에 저장
    for subdir in subdirs:
        original_subdir = tmp_map[subdir]          # 예) "A_original"
        final_subdir = os.path.join(root_dir, subdir)  # 예) "A"

        files = [f for f in os.listdir(original_subdir) if f.endswith('.png')]
        total_files = len(files)

        print(f"  - {subdir} 폴더 크롭 시작 ({total_files}개 파일)")
        for idx, file in enumerate(files, start=1):
            image_path = os.path.join(original_subdir, file)
            crop_image(image_path, final_subdir, crop_size)
            if idx % 10 == 0 or idx == total_files:
                print(f"    {subdir}: {idx}/{total_files} 파일 크롭 완료")

    # (4) 크롭이 끝난 후 원본 폴더들 삭제
    for folder in original_folders:
        shutil.rmtree(folder)
    print("  - 원본 폴더(이미지) 삭제 완료")
    print("이미지 크롭이 모두 완료되었습니다.")

def main():
    import sys

    # 실행 파일 또는 스크립트의 디렉토리 경로
    if getattr(sys, 'frozen', False):
        # PyInstaller 등으로 빌드된 실행 파일
        exe_dir = os.path.dirname(sys.executable)
    else:
        # 소스 코드 실행 시의 디렉터리
        exe_dir = os.path.dirname(os.path.abspath(__file__))

    # (1) 입력 데이터 디렉터리: exe_dir/dataset
    #     dataset 내부 구조:
    #       ├── image
    #       └── label (옵션, 존재하지 않을 수 있음)
    input_dir = os.path.join(exe_dir, 'dataset')

    # (2) 출력 데이터 디렉터리: exe_dir/input_data
    output_dir = os.path.join(exe_dir, 'input_data')

    # 경로 출력 (디버깅용)
    print(f"실행 파일 디렉터리: {exe_dir}")
    print(f"입력 디렉터리: {input_dir}")
    print(f"출력 디렉터리: {output_dir}")

    # 입력 데이터 디렉터리 존재 여부 확인
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"입력 디렉터리를 찾을 수 없습니다: {input_dir}")

    # 출력 디렉터리 생성
    create_directories(exe_dir, ['input_data'])

    # 1단계: 이미지(A, B) 및 라벨(label) 파일 매칭 후 복사 & CSV 작성
    match_files(input_dir, output_dir)

    # 2단계: 크롭 작업
    crop_images(output_dir, crop_size=256)

    print("모든 작업이 완료되었습니다.")

if __name__ == "__main__":
    main()
