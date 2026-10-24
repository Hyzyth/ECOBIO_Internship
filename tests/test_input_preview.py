import io
import json
import base64
import unittest
import cv2
import numpy as np
from flyscope.web import app


class InputPreviewTests(unittest.TestCase):
    def test_preview_shows_the_versioned_pixels_without_running_a_network(self):
        from flyscope.models.runtime import crop_rgb
        from flyscope.imaging.foreground import ISOLATED

        image = np.full((100, 100, 3), 190, np.uint8)
        cv2.ellipse(image, (50, 50), (9, 4), 0, 0, 360, (20, 30, 40), -1)
        content = cv2.imencode(".png", image)[1].tobytes()
        points = [[0, 0], [1, 0], [1, 1], [0, 1]]
        with app.test_client() as client:
            result = client.post(
                "/api/input-preview",
                data={
                    "image": (io.BytesIO(content), "image.png"),
                    "wells": json.dumps([{"id": "one", "points": points}]),
                    "mode": "fly",
                    "size": "64",
                },
            )
            self.assertEqual(result.status_code, 200)
            value = result.get_json()
            actual = cv2.imdecode(
                np.frombuffer(base64.b64decode(value["image"].split(",")[1]), np.uint8),
                cv2.IMREAD_COLOR,
            )
            expected = cv2.cvtColor(
                crop_rgb(image, points, 64, ISOLATED)[0], cv2.COLOR_RGB2BGR
            )
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(value["foreground_status"], "isolated")
            bad = client.post(
                "/api/input-preview",
                data={"image": (io.BytesIO(content), "image.png"), "wells": "[]"},
            )
            self.assertEqual(bad.status_code, 400)
