from flask import render_template, request, send_from_directory, jsonify


from backend.services.images import get_images_from_folder, extract_images_from_videos


def register_routes(app, state):

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/set-foldervideos", methods=["POST"])
    def set_foldervideos():
        path = request.get_json()["path"]
        extract_images_from_videos(path)
        return "ok"
    
    @app.route("/set-folder", methods=["POST"])
    def set_folder():
        state["image_folder"] = request.get_json()["path"]
        state["images"] = get_images_from_folder(state["image_folder"])
        return jsonify(len(state["images"]))

    @app.route("/image/<int:index>")
    def send_img(index):
        if not state["images"] or index < 0 or index >= len(state["images"]):
            return "Image introuvable", 404
        
        return send_from_directory(state["image_folder"], state["images"][index])
    

    
