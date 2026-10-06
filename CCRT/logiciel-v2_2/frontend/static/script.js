var index = 0
var indexSpecial = 0
var numImg = 0
var caseSelected = -1
var cropped = null
var scaleReduce = 10 // pour réduction taille image dans canvas p5
var compaData = null
var compaList = null

const socket = io()

	
function updateCase() {
	document.getElementById("case").innerHTML = "case sélectionnée : " + caseSelected
	if (caseSelected>=0) document.getElementById("case-state").innerHTML = "switch state index : " + grille[caseSelected].switchIndex
	updateCompaCase()
}

function updateView() {
	getImg()
	var txt = document.getElementById("status").innerHTML.trim()
	var i = txt.indexOf(" ")
	numImg = parseInt(txt.substring(i, 0))
	updateCounter()
	startP5()
	updateCase()
}

function updateCounter() {
	document.getElementById("counter").innerHTML = (index+1) + "/" + numImg
}
	
function getImg() {
	if (p5Instance) {
		p5Instance.loadNewImage(index)
	}
}

function incrementIndex() { index++ ; updateImg() }
function decrementIndex() { index-- ; updateImg() }
function checkIndex() { index = index<0 ? numImg-1 : index>=numImg ? 0 : index }

function updateImg() { checkIndex() ; getImg() ; updateCounter() }

function sq(x) { return x*x }








async function getComparaisonVal() {
	if (caseSelected < 0) return

	var c = grille[caseSelected]
	var x = c.x * scaleReduce
	var y = c.y * scaleReduce

	const res = await axios.post("/comparaison", { x: x, y: y });
	console.log("TEST aaaaaaa : ", res.data.result);
}


async function getComparaisonsVal() {
	var table = []

	grille.forEach(c => {
		table.push({x: c.x * scaleReduce, y: c.y * scaleReduce})
	})

	// const res = await axios.post("/comparaisons", table)
	// console.log("TEST bbbbbbbb : ", res.data.result)


	socket.emit("start_comparaisons", table)
}


async function readCompa() {
	const res = await axios.get("/static/comparaisons.json")
	compaData = res.data
	console.log(res.data, compaData)
}


function updateCompaCase() {
	if (caseSelected < 0 || compaData === null) return

	var data = getCompa(caseSelected)

	document.getElementById("compa-case").innerHTML = JSON.stringify(data, null, 2)
	compaList = data
}

function getCompa(caseIndex) {
	var res = []
	for (var i=0 ; i<compaData.length ; i++) {
		if (compaData[i].index == caseIndex) {
			res = compaData[i].compa
			break
		}
	}
	return res
}




function decrementIndexSpecial() { indexSpecial-- ; updateImgSpecial() }
function incrementIndexSpecial() { indexSpecial++ ; updateImgSpecial() }

function selectIndexSpecial() { index = compaList[indexSpecial] }

function checkIndexSpecial() { indexSpecial = indexSpecial<0 ? compaList.length-1 : indexSpecial>=compaList.length ? 0 : indexSpecial }
function updateImgSpecial() { checkIndexSpecial() ; selectIndexSpecial() ; getImg() ; updateCounter() }

	
function switchState() {
	if (caseSelected < 0) return
	grille[caseSelected].switchIndex = index

	updateCase()
}





async function getCnnVal() {
	var table = []

	grille.forEach((c, i) => {
		var data = {x: c.x * scaleReduce, y: c.y * scaleReduce, liste: compaData[i]}
		table.push(data)
	})


	socket.emit("start_cnn", table)
}





function downloadResults() {
	var filename = "resultats.json"

	var results = []
	grille.forEach((c, i) => {
		results.push({"index": i, "switch": c.switchIndex})
	})

	const json = JSON.stringify(results, null, 2);
    const blob = new Blob([json], { type: "application/json" });

    const url = URL.createObjectURL(blob);

    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();

    URL.revokeObjectURL(url);
}




socket.on("server_message_comparaisons", (data) => {
	console.log("socketio : ", data)

	switch(data.status) {
		case 1: document.getElementById("compas_status").innerHTML = "calculs en cours"; break
		case 2: document.getElementById("compas_status").innerHTML = "fin calculs"; break
	}

	if (data.status == 2) {
		compaData = data.result
		updateCompaCase()
	}

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
			grille[i].switchIndex = elt.cnn
		})
	}

	// if (data.status == 2) {
	// 	compaData = data.result
	// 	updateCompaCase()
	// }

})