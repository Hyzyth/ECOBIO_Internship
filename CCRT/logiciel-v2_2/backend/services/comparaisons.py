import os
import json
import cv2
import numpy as np
from backend.services.images import crop_image2

def compare_images(img1, img2):
    # gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(img1, img2)
    diff_blur = cv2.GaussianBlur(diff, (5,5), 0)
    _, thresh = cv2.threshold(diff_blur, 30, 255, cv2.THRESH_BINARY)

    return np.count_nonzero(thresh)


def run_comparaisons(data, image_folder, images, socket, sid):
    results = []
    loaded_images = []

    for image_name in images:
        path = f"{image_folder}/{image_name}"

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"[ERROR] Impossible de charger : {path}")
            continue

        loaded_images.append(img)


    for idx, cell in enumerate(data):
        x = int(cell["x"])
        y = int(cell["y"])

        res = []

        prev = None
        for i in range(len(images)-1):
            # path = f"{image_folder}/{images[i]}"
            # curr = crop_image(path, x, y, out_size=20)
            curr = crop_image2(loaded_images[i], x, y, out_size=50)

            if prev is not None:
                try:
                    if compare_images(prev, curr) > 0:
                        res.append(i)
                except Exception as e:
                    print(f"[ERROR] index {i} : {e}")
                    continue

            prev = curr

        results.append({"index": idx, "compa": res})
        print("COUNTER COMPA :", idx)

        prog = (idx / len(data)) * 100

        socket.emit("server_message_comparaisons_progress",
                           {"result": prog},
                           to=sid)


    # save_json_to_static(results)

    return results



def save_json_to_static(data):
	filename = "comparaisons.json"
	file_path = os.path.join("../frontend/static/", filename)

	with open(file_path, "w", encoding="utf-8") as f:
		json.dump(data, f, indent=2, ensure_ascii=False)

	return file_path