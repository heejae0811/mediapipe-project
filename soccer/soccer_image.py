import cv2
import os

def extract_frames(video_path, output_folder='frames'):
    """
    영상에서 모든 프레임을 이미지로 추출하여 저장

    Parameters:
    video_path (str): 영상 파일 경로
    output_folder (str): 프레임 이미지를 저장할 폴더명
    """

    # 출력 폴더가 없으면 생성
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 영상 파일 열기
    video = cv2.VideoCapture(video_path)

    # 영상이 제대로 열렸는지 확인
    if not video.isOpened():
        print("영상 파일을 열 수 없습니다.")
        return

    # 영상 정보 출력
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"FPS: {fps}")
    print(f"총 프레임 수: {total_frames}")

    frame_count = 0

    while True:
        # 프레임 읽기
        ret, frame = video.read()

        # 더 이상 프레임이 없으면 종료
        if not ret:
            break

        # 프레임을 이미지 파일로 저장
        frame_filename = os.path.join(output_folder, f'frame_{frame_count:06d}.jpg')
        cv2.imwrite(frame_filename, frame)

        frame_count += 1

        # 진행 상황 출력
        if frame_count % 100 == 0:
            print(f"처리 중... {frame_count}/{total_frames} 프레임")

    # 리소스 해제
    video.release()
    print(f"\n완료! 총 {frame_count}개의 프레임을 '{output_folder}' 폴더에 저장했습니다.")


def extract_frames_by_interval(video_path, output_folder='frames', interval=1):
    """
    영상에서 일정 간격으로 프레임을 추출하여 저장

    Parameters:
    video_path (str): 영상 파일 경로
    output_folder (str): 프레임 이미지를 저장할 폴더명
    interval (int): 프레임 추출 간격 (1=모든 프레임, 2=2프레임마다 1개, 30=1초마다 1개(30fps 기준))
    """

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print("영상 파일을 열 수 없습니다.")
        return

    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"FPS: {fps}")
    print(f"총 프레임 수: {total_frames}")
    print(f"추출 간격: {interval}프레임마다")

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = video.read()

        if not ret:
            break

        # 지정된 간격으로만 저장
        if frame_count % interval == 0:
            frame_filename = os.path.join(output_folder, f'frame_{saved_count:06d}.jpg')
            cv2.imwrite(frame_filename, frame)
            saved_count += 1

            if saved_count % 50 == 0:
                print(f"처리 중... {saved_count}개 저장됨")

        frame_count += 1

    video.release()
    print(f"\n완료! 총 {saved_count}개의 프레임을 '{output_folder}' 폴더에 저장했습니다.")


# 사용 예시
if __name__ == "__main__":
    # 방법 1: 모든 프레임 추출
    extract_frames('kick2.MOV', 'output_frames')

    # 방법 2: 30프레임마다 1개씩 추출 (1초에 1장, 30fps 기준)
    # extract_frames_by_interval('your_video.mp4', 'output_frames', interval=30)