var flagMouse = false


class Cible {
	constructor(x, y, color) {
		this.color = color
		this.d = 20
		this.x = x
		this.y = y
	}

	draw(p) {
		p.noFill()
		p.strokeWeight(3)
		p.stroke(this.color)
		p.circle(this.x, this.y, this.d)

		if ( (sq(this.d) >= sq(this.x - p.mouseX) + sq(this.y - p.mouseY)) && p.mouseIsPressed ) {
			this.x = p.mouseX
			this.y = p.mouseY
		}
	}
}

class Case {
	constructor(x, y, index) {
		this.selected = false
		this.switchIndex = -1
		this.x = x
		this.y = y
		this.d = 20
		this.index = index
	}

	draw(p) {
		p.noFill()
		p.strokeWeight(1)
		if (this.selected) {
			p.stroke("cyan")
		} else {
			if (this.switchIndex > 0) {
				p.stroke("orange")
			} else {
				p.stroke("blue")
			}
		}
		p.rect(this.x-this.d/2, this.y-this.d/2, this.d)


		if ( (sq(this.d) >= sq(this.x - p.mouseX) + sq(this.y - p.mouseY)) && p.mouseIsPressed && flagMouse) {
			selectCase(this.index)
			flagMouse = false
		}
	}
}





var cibleO = new Cible(50, 50, "blue")
var cibleX = new Cible(200, 50, "red")
var cibleY = new Cible(50, 200, "green")



function createGrille() {
	state.grille = []
	
	var nx = 6
	var ny = 6

	var lx = (cibleX.x - cibleO.x) / (nx-1)
	var ly = (cibleY.y - cibleO.y) / (ny-1)
	
	k = 0

	for (var i=0 ; i<nx ; i++) {
		for (var j=0 ; j<ny ; j++) {
			var c = new Case(cibleO.x+lx*i , cibleO.y+ly*j, k)
			k++
			state.grille.push(c)
		}
	}

	for (var i=0 ; i<nx-1 ; i++) {
		for (var j=0 ; j<ny-1 ; j++) {
			var c = new Case(lx*1/2 + cibleO.x+lx*i , ly*1/2 + cibleO.y+ly*j, k)
			k++
			state.grille.push(c)
		}
	}
}



function startP5() {

	if (state.p5Instance) {
		state.p5Instance.remove()
		state.p5Instance = null
	}

	
	state.p5Instance = new p5((p) => {
		let img
		let img_prev
		let img_next

		p.loadNewImage = (index) => {
			img = p.loadImage(`/image/${index}?t=${Date.now()}`)
			img_prev = p.loadImage(`/image/${index-1}?t=${Date.now()}`)
			img_next = p.loadImage(`/image/${index+1}?t=${Date.now()}`)

			// console.log(img, img_next, img_prev)
		}

		p.setup = () => {
			p.createCanvas(406, 304+100+50).parent("p5-container")
			if (state.numImg > 0) p.loadNewImage(state.index)
		}

		p.draw = () => {
			if (!img) return

			var W = img.width
			var H = img.height
			p.image(img, 0, 0, W/state.scaleReduce, H/state.scaleReduce)

			if (state.caseSelected >= 0) {
				var c = state.grille[state.caseSelected]

				var x = c.x * state.scaleReduce
				var y = c.y * state.scaleReduce
				var w = 200
				var h = 200

				let cropped = img.get(x-w/2, y-h/2, w, h)
				let cropped_prev = img_prev.get(x-w/2, y-h/2, w, h)
				let cropped_next = img_next.get(x-w/2, y-h/2, w, h)


				p.image(cropped_prev, 0, 304+10, 100, 100)
				p.image(cropped, 120, 304+10, 100, 100)
				p.image(cropped_next, 240, 304+10, 100, 100)


				// p.noStroke()
				// p.fill("black")
				// p.text("img " + (state.index-1).toString(), 50,304+10+110)
				// p.text("img " + (state.index).toString(), 50+120,304+10+110)
				// p.text("img " + (state.index+1).toString(), 50+240,304+10+110)
			}


			cibleO.draw(p)
			cibleX.draw(p)
			cibleY.draw(p)


			state.grille.forEach(c => {
				c.draw(p)
			})
		}

		p.mouseReleased = () => {
			flagMouse = true
		}
	})
}


startP5()