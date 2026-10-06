import os
import sys
import webbrowser
from flask import Flask      
from flask_socketio import SocketIO

from backend.routes.routes  import register_routes
from backend.routes.sockets import register_socket_events




def resource_path(relative_path):
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))



app = Flask(__name__, template_folder="./frontend/templates", static_folder="./frontend/static")
socketio = SocketIO(app, async_mode="threading")


state = {
	"images": [],
	"image_index": 0,
	"image_folder": None,
}


register_routes(app, state)
register_socket_events(socketio, state)



DATA_FOLDER = os.path.join(os.path.expanduser("~"), "Logiciel_runtime")
os.makedirs(DATA_FOLDER, exist_ok=True)


if __name__ == "__main__" :
	webbrowser.open("http://127.0.0.1:5000")
	socketio.run(app, host="0.0.0.0", port=5000)
	



