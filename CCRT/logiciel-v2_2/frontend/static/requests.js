async function setFolder_videos() {
	const folder = document.getElementById("foldervideos_input").value.trim()
	const res = await axios.post("/set-foldervideos", {path: folder})
	document.getElementById("foldervideos_status").innerHTML = "dossier images créé"
}


async function setFolder() {
	const folder = document.getElementById("folder_input").value.trim()
	const res = await axios.post("/set-folder", {path: folder})
	state.numImg = res.data
	document.getElementById("folder_status").innerHTML = res.data + " images dans dossier"

	updateCanvas()
}


async function readCompa() {
	const res = await axios.get("/static/comparaisons.json")
	state.compaData = res.data
	console.log(state.compaData)
}


async function getCnnVal() {
	var table = []

	state.grille.forEach((c, i) => {
		var data = {x: c.x * state.scaleReduce, y: c.y * state.scaleReduce, liste: state.compaData[i]}
		table.push(data)
	})

	socket.emit("start_cnn", table)
}