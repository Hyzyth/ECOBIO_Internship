from flask import request

from backend.services.comparaisons import run_comparaisons
from backend.services.cnn import predict_image, run_cnn


def register_socket_events(socketio, state):

    @socketio.on("start_comparaisons")
    def start_comparaisons(data):
        sid = request.sid
        socketio.emit("server_message_comparaisons", {"status": 1}, to=sid)

        def task():
            result = run_comparaisons(data, state["image_folder"], state["images"], socketio, sid)
            socketio.emit("server_message_comparaisons",
                           {"status": 2, "result": result},
                           to=sid)

        socketio.start_background_task(task)

    

    @socketio.on("start_cnn")
    def start_cnn(data):
        sid = request.sid
        socketio.emit("server_message_cnn", {"status": 1}, to=sid)

        def task():
            result = run_cnn(data, state["image_folder"], state["images"], socketio, sid)
            socketio.emit("server_message_cnn",
                           {"status": 2, "result": result},
                           to=sid)

        socketio.start_background_task(task)
