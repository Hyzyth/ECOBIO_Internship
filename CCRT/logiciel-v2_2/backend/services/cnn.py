import cv2
import numpy as np
from tensorflow import keras

from backend.services.images import crop_image


path_model = "backend/models/model1.keras"
model = keras.models.load_model(path_model)

img_width = 200
img_height = 200

class_names = ["coma", "awake"]



def preprocess_fly_image(img):
    alpha, beta = 1.5, 120
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def predict_image(model, img, img_width, img_height, class_names):
    img = preprocess_fly_image(img)
    img = cv2.resize(img, (img_width, img_height))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=(0, -1))

    pred = model.predict(img)
    idx = np.argmax(pred, axis=1)[0]
    return pred[0], class_names[idx]



def run_cnn(data, image_folder, images, socket, sid):
	results = []

	m = len(data)

	for j in range(m):
		cell = data[j]
		x = int(cell["x"])
		y = int(cell["y"])
		liste = cell["liste"]["compa"]
		n = len(liste)

		results.append({"index": j, "cnn": -1})

		for i in liste:
			path = str(image_folder) + "/" + str(images[i])
			tampon3 = crop_image(path, x, y, out_size=250)
			probabilities, predicted_class = predict_image(model, tampon3, img_width, img_height, class_names)
			print("CNN = ", "case :", j, " - img :", i, " - results :", probabilities, predicted_class)
			if (predicted_class == "awake"):
				results[j]["cnn"] = i
				break

		prog = (j / m) * 100

		socket.emit("server_message_cnn_progress",
                           {"result": prog},
                           to=sid)

	return results
