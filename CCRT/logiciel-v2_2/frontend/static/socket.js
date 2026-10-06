
const socket = io()

async function getComparaisonsVal() {
	var table = []

	state.grille.forEach(c => {
		table.push({x: c.x * state.scaleReduce, y: c.y * state.scaleReduce})
	})

	socket.emit("start_comparaisons", table)
}



socket.on("server_message_comparaisons", (data) => {
	console.log("socketio : ", data)

	switch(data.status) {
		case 1: document.getElementById("compas_status").innerHTML = "calculs en cours"; break
		case 2: document.getElementById("compas_status").innerHTML = "fin calculs"; break
	}

	if (data.status == 2) {
		state.compaData = data.result
		updateCompaCase()

		document.getElementById("compas_prog").innerHTML = 100;
	}

})

socket.on("server_message_comparaisons_progress", (data) => {
	console.log("socketio : ", data)
	var prog = parseInt(data.result)
	document.getElementById("compas_prog").innerHTML = prog;
	
})



socket.on("server_message_cnn", (data) => {
	console.log("socketio : ", data)

	switch(data.status) {
		case 1: document.getElementById("cnn_status").innerHTML = "calculs en cours"; break
		case 2: document.getElementById("cnn_status").innerHTML = "fin calculs"; break
	}

	if (data.status == 2) {
		console.log("TEST99", data.result)
		data.result.forEach((elt, i) => {
			state.grille[i].switchIndex = elt.cnn
		})

		document.getElementById("cnn_prog").innerHTML = 100;
	}
})


socket.on("server_message_cnn_progress", (data) => {
	console.log("socketio : ", data)
	var prog = parseInt(data.result)
	document.getElementById("cnn_prog").innerHTML = prog;
	
})


