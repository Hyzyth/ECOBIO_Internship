import os
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta



def get_images_from_folder(folder_path):
    if not os.path.isdir(folder_path):
        return []
    valid_extensions = (".png", ".jpg", ".jpeg", ".bmp")
    return sorted(f for f in os.listdir(folder_path)
                  if f.lower().endswith(valid_extensions))


def crop_image(src_path, x, y, crop_size=250, out_size=50):
    img = cv2.imread(src_path)
    if img is None:
        raise ValueError("Image non chargée")

    h, w = img.shape[:2]
    x2 = max(0, min(int(x - crop_size / 2), w - crop_size))
    y2 = max(0, min(int(y - crop_size / 2), h - crop_size))

    cropped = img[y2:y2+crop_size, x2:x2+crop_size]
    return cv2.resize(cropped, (out_size, out_size), cv2.INTER_AREA)


def crop_image2(img, x, y, crop_size=250, out_size=50):
    if img is None:
        raise ValueError("Image non chargée")

    h, w = img.shape[:2]
    x2 = max(0, min(int(x - crop_size / 2), w - crop_size))
    y2 = max(0, min(int(y - crop_size / 2), h - crop_size))

    cropped = img[y2:y2+crop_size, x2:x2+crop_size]
    return cv2.resize(cropped, (out_size, out_size), cv2.INTER_AREA)



def extract_images_from_videos(root_folder, frame_interval_sec=1):

    root_path = Path(root_folder)

    videos_folder = root_path / "videos"

    if not videos_folder.exists() or not videos_folder.is_dir():
        print("Le dossier 'videos' est introuvable.")
        return

    images_folder = root_path / "images"
    images_folder.mkdir(exist_ok=True)

    supported_extensions = [".h264", ".mp4", ".avi", ".mov"]

    video_files = [
        file for file in videos_folder.iterdir()
        if file.suffix.lower() in supported_extensions
    ]

    if not video_files:
        print("Aucune vidéo trouvée.")
        return

    for video_path in video_files:

        print(f"\nTraitement : {video_path.name}")

        creation_datetime = datetime.fromtimestamp(
            video_path.stat().st_mtime
        )

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            print(f"Impossible d'ouvrir : {video_path.name}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)

        if fps <= 0:
            print(f"FPS invalide : {video_path.name}")
            cap.release()
            continue

        frame_interval = int(fps * frame_interval_sec)

        frame_index = 0

        while True:
            success, frame = cap.read()

            if not success:
                break

            if frame_index % frame_interval == 0:

                second_offset = int(frame_index / fps)

                image_datetime = (
                    creation_datetime +
                    timedelta(seconds=second_offset)
                )

                filename = (
                    image_datetime.strftime("%Y-%m-%d_%H-%M-%S")
                    + ".jpg"
                )

                output_path = images_folder / filename

                cv2.imwrite(str(output_path), frame)

                print(f"Sauvegardé : {filename}")

            frame_index += 1

        cap.release()

    print("\nExtraction terminée.")