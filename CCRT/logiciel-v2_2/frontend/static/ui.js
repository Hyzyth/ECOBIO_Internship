function updateCanvas() {
	state.p5Instance.loadNewImage(state.index)
	updateCounter()
	updateCase()
}

function updateCounter() { document.getElementById("counter").innerHTML = (state.index) + "/" + state.numImg }


function incrementIndex() { state.index++ ; checkIndex() ; updateCounter() ; updateImg() }
function decrementIndex() { state.index-- ; checkIndex() ; updateCounter() ; updateImg() }
function checkIndex() { state.index = state.index<0 ? state.numImg-1 : state.index>=state.numImg ? 0 : state.index }
function updateCounter() { document.getElementById("counter").innerHTML = (state.index) + "/" + state.numImg }

function updateImg() { if (state.p5Instance) state.p5Instance.loadNewImage(state.index)  }




function selectCase(k) {
	var c = state.grille[k]
	if (c.selected) c.selected = false
	else {
		if (state.caseSelected >= 0) state.grille[state.caseSelected].selected = false
		state.grille[k].selected = true
		state.caseSelected = k
	}

	updateCase()
}

function updateCase() {
	document.getElementById("case").innerHTML = "case sélectionnée : " + state.caseSelected
	if (state.caseSelected>=0) document.getElementById("case-state").innerHTML = "switch state index : " + state.grille[state.caseSelected].switchIndex
	//updateCompaCase()
}



function updateCompaCase() {
	if (state.caseSelected < 0 || state.compaData === null) return

	var data = getCompa(state.caseSelected)

	document.getElementById("compa-case").innerHTML = JSON.stringify(data, null, 2)
	state.compaList = data
}

function getCompa(caseIndex) {
	var res = []
	for (var i=0 ; i<state.compaData.length ; i++) {
		if (state.compaData[i].index == caseIndex) {
			res = state.compaData[i].compa
			break
		}
	}
	return res
}



function decrementIndexSpecial() { state.indexSpecial-- ; updateImgSpecial() }
function incrementIndexSpecial() { state.indexSpecial++ ; updateImgSpecial() }

function selectIndexSpecial() { state.index = state.compaList[state.indexSpecial] }

function checkIndexSpecial() { state.indexSpecial = state.indexSpecial<0 ? state.compaList.length-1 : state.indexSpecial>=state.compaList.length ? 0 : state.indexSpecial }
function updateImgSpecial() { checkIndexSpecial() ; selectIndexSpecial() ; updateImg() ; updateCounter() }

	
function switchState() {
	if (state.caseSelected < 0) return
	state.grille[state.caseSelected].switchIndex = state.index
	updateCase()
}



function downloadResults() {
	var filename = "resultats.json"

	var results = []
	state.grille.forEach((c, i) => {
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
