package render

import (
	"github.com/lucasb-eyer/go-colorful"
	"image"
	"image/color"
	"image/draw"
	_ "image/png"
	"log"
	"math/big"
	"math/cmplx"
	"sync"
)

type FrameInfo struct {
	boundary float64
	xmin float64
	ymin float64
	xmax float64
	ymax float64
	centerX float64
	centerY float64
}

func ConstructFrameInfo(b, xmin, ymin, xmax, ymax, cx, cy float64) FrameInfo {
	return FrameInfo {
		boundary: b,
		xmin: xmin,
		ymin: ymin,
		xmax: xmax,
		ymax: ymax,
		centerX: cx,
		centerY: cy,
	}
}

func (f FrameInfo) Read() (
	float64,
	float64,
	float64,
	float64,
	float64,
	float64,
	float64,
) {
	return f.boundary, f.xmin, f.ymin, f.xmax, f.ymax, f.centerX, f.centerY
}

type FrameInfoHP struct {
	boundary *big.Float
	xmin     *big.Float
	ymin     *big.Float
	xmax     *big.Float
	ymax     *big.Float
	centerX  *big.Float
	centerY  *big.Float
}

func ConstructFrameInfoHP(b, xmin, ymin, xmax, ymax, cx, cy *big.Float) FrameInfoHP {
	return FrameInfoHP{
		boundary: b,
		xmin:     xmin,
		ymin:     ymin,
		xmax:     xmax,
		ymax:     ymax,
		centerX:  cx,
		centerY:  cy,
	}
}

func (f FrameInfoHP) Read() (
	*big.Float,
	*big.Float,
	*big.Float,
	*big.Float,
	*big.Float,
	*big.Float,
	*big.Float,
) {
	return f.boundary, f.xmin, f.ymin, f.xmax, f.ymax, f.centerX, f.centerY
}

func combine(
	width, height int, c1, c2, c3, c4 <-chan image.Image,
) <-chan image.Image {
	c := make(chan image.Image)
	go func() {
		var wg sync.WaitGroup
		newImage := image.NewRGBA(image.Rect(0, 0, width, height))

		copy := func(
			dst draw.Image,
			r image.Rectangle,
			src image.Image,
			sp image.Point,
		) {
			draw.Draw(dst, r, src, sp, draw.Src)
			wg.Done()
		}

		wg.Add(4)
		var s1, s2, s3, s4 image.Image
		var ok1, ok2, ok3, ok4 bool

		topLeft := image.Rect(0, 0, width/2, height/2)
		topRight := image.Rect(width/2, 0, width, height/2)
		botLeft := image.Rect(0, height/2, width/2, height)
		botRight := image.Rect(width/2, height/2, width, height)

		for {
			select {
			case s1, ok1 = <-c1:
				go copy(newImage, topLeft, s1, image.Point{0, 0})
			case s2, ok2 = <-c2:
				go copy(newImage, topRight, s2, image.Point{0, 0})
			case s3, ok3 = <-c3:
				go copy(newImage, botLeft, s3, image.Point{0, 0})
			case s4, ok4 = <-c4:
				go copy(newImage, botRight, s4,
					image.Point{0, 0})
			}
			if ok1 && ok2 && ok3 && ok4 {
				break
			}
		}
		wg.Wait()
		c <- newImage
	}()
	return c
}

func BigPrint(num *big.Float) string {
	return num.Text('g', -1)
}

type MandelFunc func(complex128) color.Color

func GetMandelFunc(colorized bool) MandelFunc {
	if colorized {
		return mandelbrotColor
	}
	return mandelbrotMonochrome
}

func mandelbrotMonochrome(z complex128) color.Color {
	const iterations = 100
	const contrast = 15

	var v complex128
	for n := uint8(0); n < iterations; n++ {
		v = v*v + z
		if cmplx.Abs(v) > 2 {
			return color.Gray{255 - contrast*n}
		}
	}
	return color.Black
}

func mandelbrotColor(z complex128) color.Color {
	const iterations = 100
	const contrast = 15

	var v complex128
	for n := uint8(0); n < iterations; n++ {
		v = v*v + z
		if cmplx.Abs(v) > 2 {
			return colorful.Hsv(float64(contrast*n), 50, 100)
		}
	}
	return color.Black
}

func mandelbrotFloat(zR, zI *big.Float) color.Color {
	const iterations = 100
	const contrast = 15

	vR := new(big.Float)
	vI := new(big.Float)
	for n := uint8(0); n < iterations; n++ {
		// v = v*v + z
		// (r+i)^2=r^2 + 2ri + i^2
		vR2, vI2 := new(big.Float), new(big.Float)
		vR2.Mul(vR, vR).Sub(vR2, new(big.Float).Mul(vI, vI)).Add(vR2, zR)
		vI2.Mul(vR, vI).Mul(vI2, big.NewFloat(2)).Add(vI2, zI)
		vR, vI = vR2, vI2

		squareSum := new(big.Float)
		squareSum.Mul(vR, vR).Add(squareSum, new(big.Float).Mul(vI, vI))
		if squareSum.Cmp(big.NewFloat(4)) > 0 {
			return colorful.Hsv(float64(contrast*n), 50, 100)
		}
	}
	return color.Black
}

/*
x1, y1 | x2, y1
x1, y2 | x2, y2
*/
func generateSubpixelCoords(x, y, stepSize float64) [][]float64 {
	ret := make([][]float64, 4)
	for i := 0; i < 4; i++ {
		ret[i] = make([]float64, 2)
	}

	ret[0][0] = x - stepSize
	ret[0][1] = y - stepSize

	ret[1][0] = x + stepSize
	ret[1][1] = y + stepSize

	ret[2][0] = x + stepSize
	ret[2][1] = y - stepSize

	ret[3][0] = x - stepSize
	ret[3][1] = y + stepSize
	return ret
}

func getAverageMandelbrot(coords [][]float64, m MandelFunc) color.Color {
	allColors := make([]color.Color, 0)
	for i := 0; i < len(coords); i++ {
		x, y := coords[i][0], coords[i][1]
		z := complex(x, y)
		allColors = append(allColors, m(z))
	}

	var rSum uint32
	var gSum uint32
	var bSum uint32
	for _, col := range allColors {
		r, g, b, _ := col.RGBA()
		rSum += r
		gSum += g
		bSum += b
	}

	numberOfColors := uint32(len(allColors))
	rAvg := uint8(rSum / numberOfColors)
	gAvg := uint8(gSum / numberOfColors)
	bAvg := uint8(bSum / numberOfColors)
	return color.NRGBA{rAvg, gAvg, bAvg, 255}
}

func renderMBoundsAA(
	width, height int,
	xmin, ymin, xmax, ymax float64,
	m MandelFunc,
) <-chan image.Image {
	log.Printf("rendering bounds (%f, %f), (%f, %f)\n", xmin, ymin, xmax, ymax)
	c := make(chan image.Image)
	stepSize := (xmax - xmin) / float64(width)
	go func() {
		img := image.NewRGBA(image.Rect(0, 0, width, height))
		for py := 0; py < height; py++ {
			y := float64(py)/float64(height)*(ymax-ymin) + ymin
			for px := 0; px < width; px++ {
				x := float64(px)/float64(width)*(xmax-xmin) + xmin
				subs := generateSubpixelCoords(x, y, stepSize/2)
				// Image point (px, py) represents complex value z.
				img.Set(px, py, getAverageMandelbrot(subs, m))
			}
		}
		c <- img
	}()

	return c
}

func RenderMFrameAA(
	width, height int,
	f FrameInfo,
	m MandelFunc,
) <-chan image.Image {
	boundary, xmin, ymin, _, _, cx, cy := f.Read()
	c1 := renderMBoundsAA(
		width/2,
		height/2,
		xmin,
		ymin,
		xmin+boundary,
		ymin+boundary,
		m,
	)
	c2 := renderMBoundsAA(
		width/2,
		height/2,
		cx,
		ymin,
		cx+boundary,
		ymin+boundary,
		m,
	)
	c3 := renderMBoundsAA(
		width/2,
		height/2,
		xmin,
		cy,
		xmin+boundary,
		cy+boundary,
		m,
	)
	c4 := renderMBoundsAA(
		width/2,
		height/2,
		cx,
		cy,
		cx+boundary,
		cy+boundary,
		m,
	)
	return combine(width, height, c1, c2, c3, c4)
}

func renderMBoundsHP(
	width, height int, xmin, ymin, xmax, ymax *big.Float,
) <-chan image.Image {
	log.Printf("rendering bounds (%s, %s), (%s, %s)\n",
		BigPrint(xmin), BigPrint(ymin), BigPrint(xmax), BigPrint(ymax))
	c := make(chan image.Image)
	go func() {
		img := image.NewRGBA(image.Rect(0, 0, width, height))
		for py := 0; py < height; py++ {
			//y := float64(py)/float64(height)*(ymax-ymin) + ymin
			y := new(big.Float).Quo(big.NewFloat(float64(py)), big.NewFloat(float64(height)))
			diff := new(big.Float).Sub(ymax, ymin)
			y.Mul(y, diff).Add(y, ymin)
			for px := 0; px < width; px++ {
				//x := float64(px)/float64(width)*(xmax-xmin) + xmin
				x := new(big.Float).Quo(big.NewFloat(float64(px)), big.NewFloat(float64(width)))
				diff := new(big.Float).Sub(xmax, xmin)
				x.Mul(x, diff).Add(x, xmin)
				// Image point (px, py) represents complex value z.
				img.Set(px, py, mandelbrotFloat(x, y))
			}
		}
		c <- img
	}()

	return c
}

// M stands for mandelbrot
// HP stands for high-precision.
func RenderMFrameHP(width, height int, f FrameInfoHP) <-chan image.Image {
	boundary, xmin, ymin, _, _, cx, cy := f.Read()
	c1 := renderMBoundsHP(
		width/2,
		height/2,
		xmin,
		ymin,
		new(big.Float).Add(xmin, boundary),
		new(big.Float).Add(ymin, boundary),
	)
	c2 := renderMBoundsHP(
		width/2,
		height/2,
		cx,
		ymin,
		new(big.Float).Add(cx, boundary),
		new(big.Float).Add(ymin, boundary),
	)
	c3 := renderMBoundsHP(
		width/2,
		height/2,
		xmin,
		cy,
		new(big.Float).Add(xmin, boundary),
		new(big.Float).Add(cy, boundary),
	)
	c4 := renderMBoundsHP(
		width/2,
		height/2,
		cx,
		cy,
		new(big.Float).Add(cx, boundary),
		new(big.Float).Add(cy, boundary),
	)
	return combine(width, height, c1, c2, c3, c4)
}

func renderMBounds(
	width, height int,
	xmin, ymin, xmax, ymax float64,
	m MandelFunc,
) <-chan image.Image {
	log.Printf("rendering bounds (%f, %f), (%f, %f)\n", xmin, ymin, xmax, ymax)
	c := make(chan image.Image)
	go func() {
		img := image.NewRGBA(image.Rect(0, 0, width, height))
		for py := 0; py < height; py++ {
			y := float64(py)/float64(height)*(ymax-ymin) + ymin
			for px := 0; px < width; px++ {
				x := float64(px)/float64(width)*(xmax-xmin) + xmin
				// Image point (px, py) represents complex value z.
				img.Set(px, py, m(complex(x, y)))
			}
		}
		c <- img
	}()

	return c
}

func RenderMFrame(
	width, height int, f FrameInfo, m MandelFunc,
) <-chan image.Image {
	boundary, xmin, ymin, _, _, cx, cy := f.Read()
	c1 := renderMBounds(
			width/2,
			height/2,
			xmin,
			ymin,
			xmin + boundary,
			ymin + boundary,
			m,
		)
	c2 := renderMBounds(
			width/2,
			height/2,
			cx,
			ymin,
			cx + boundary,
			ymin + boundary,
			m,
		)
	c3 := renderMBounds(
			width/2,
			height/2,
			xmin,
			cy,
			xmin + boundary,
			cy + boundary,
			m,
		)
	c4 := renderMBounds(
			width/2,
			height/2,
			cx,
			cy,
			cx + boundary,
			cy + boundary,
			m,
		)
	return combine(width, height, c1, c2, c3, c4)
}

// Newton fractals

type validFunc func(complex128) complex128
type NewtonFunc func(complex128) color.Color

func getAverageNewton(coords [][]float64, n NewtonFunc) color.Color {
	allColors := make([]color.Color, 0)
	for i := 0; i < len(coords); i++ {
		x, y := coords[i][0], coords[i][1]
		z := complex(x, y)
		allColors = append(allColors, n(z))
	}

	var rSum uint32
	var gSum uint32
	var bSum uint32
	for _, col := range allColors {
		r, g, b, _ := col.RGBA()
		rSum += r
		gSum += g
		bSum += b
	}

	numberOfColors := uint32(len(allColors))
	rAvg := uint8(rSum / numberOfColors)
	gAvg := uint8(gSum / numberOfColors)
	bAvg := uint8(bSum / numberOfColors)
	return color.NRGBA{rAvg, gAvg, bAvg, 255}
}

func renderNBoundsAA(
	width, height int,
	xmin, ymin, xmax, ymax float64,
	n NewtonFunc,
) <-chan image.Image {
	log.Printf("rendering bounds (%f, %f), (%f, %f)\n", xmin, ymin, xmax, ymax)
	c := make(chan image.Image)
	stepSize := (xmax - xmin) / float64(width)
	go func() {
		img := image.NewRGBA(image.Rect(0, 0, width, height))
		for py := 0; py < height; py++ {
			y := float64(py)/float64(height)*(ymax-ymin) + ymin
			for px := 0; px < width; px++ {
				x := float64(px)/float64(width)*(xmax-xmin) + xmin
				subs := generateSubpixelCoords(x, y, stepSize/2)
				// Image point (px, py) represents complex value z.
				img.Set(px, py, getAverageNewton(subs, n))
			}
		}
		c <- img
	}()

	return c
}

func RenderNFrameAA(
	width, height int,
	f FrameInfo,
	n NewtonFunc,
) <-chan image.Image {
	boundary, xmin, ymin, _, _, cx, cy := f.Read()
	c1 := renderNBoundsAA(
		width/2,
		height/2,
		xmin,
		ymin,
		xmin+boundary,
		ymin+boundary,
		n,
	)
	c2 := renderNBoundsAA(
		width/2,
		height/2,
		cx,
		ymin,
		cx+boundary,
		ymin+boundary,
		n,
	)
	c3 := renderNBoundsAA(
		width/2,
		height/2,
		xmin,
		cy,
		xmin+boundary,
		cy+boundary,
		n,
	)
	c4 := renderNBoundsAA(
		width/2,
		height/2,
		cx,
		cy,
		cx+boundary,
		cy+boundary,
		n,
	)
	return combine(width, height, c1, c2, c3, c4)
}

func renderNBounds(
	width, height int,
	xmin, ymin, xmax, ymax float64,
	n NewtonFunc,
) <-chan image.Image {
	log.Printf("rendering bounds (%f, %f), (%f, %f)\n", xmin, ymin, xmax, ymax)

	c := make(chan image.Image)
	go func() {
		img := image.NewRGBA(image.Rect(0, 0, width, height))
		for py := 0; py < height; py++ {
			y := float64(py)/float64(height)*(ymax-ymin) + ymin
			for px := 0; px < width; px++ {
				x := float64(px)/float64(width)*(xmax-xmin) + xmin
				// Image point (px, py) represents complex value z.
				img.Set(px, py, n(complex(x, y)))
			}
		}
		c <- img
	}()

	return c
}

func RenderNFrame(
	width, height int, f FrameInfo, n NewtonFunc,
) <-chan image.Image {
	boundary, xmin, ymin, _, _, cx, cy := f.Read()
	c1 := renderNBounds(
			width/2,
			height/2,
			xmin,
			ymin,
			xmin + boundary,
			ymin + boundary,
			n,
		)
	c2 := renderNBounds(
			width/2,
			height/2,
			cx,
			ymin,
			cx + boundary,
			ymin + boundary,
			n,
		)
	c3 := renderNBounds(
			width/2,
			height/2,
			xmin,
			cy,
			xmin + boundary,
			cy + boundary,
			n,
		)
	c4 := renderNBounds(
			width/2,
			height/2,
			cx,
			cy,
			cx + boundary,
			cy + boundary,
			n,
		)
	return combine(width, height, c1, c2, c3, c4)
}

func newtonMonochrome(
	z, a complex128,
	function, derivative validFunc,
) color.Color {
	const iterations = 200
	const contrast = 15

	for n := uint8(0); n < iterations; n++ {
		numerator := function(z)
		z = z - a*(numerator/derivative(z))
		if cmplx.Abs(numerator) < 0.001 {
			return color.Gray{255 - contrast*n}
		}
	}
	return color.Black
}

func newtonColor(
	z, a complex128,
	function, derivative validFunc,
) color.Color {
	const iterations = 200
	const contrast = 15

	for n := uint8(0); n < iterations; n++ {
		numerator := function(z)
		z = z - a*(numerator/derivative(z))
		if cmplx.Abs(numerator) < 0.001 {
			return colorful.Hsv(float64(contrast*n), 50, 100)
		}
	}
	return color.Black
}

// f(z) = z^4 - 1
// f'(z) = 4z^3
func NewtonOne(inColor bool) NewtonFunc {
	f := func(x complex128) complex128 {
		return x*x*x*x - 1
	}
	d := func (x complex128) complex128 {
		return 4*x*x*x
	}
	a := complex(1.0, 0)
	if inColor {
		return func(z complex128)  color.Color {
			return newtonColor(z, a, f, d)
		}
	}
	return func(z complex128) color.Color {
		return newtonMonochrome(z, a, f, d)
	}
}

// f(z) = z^3 - 1
// f'(z) = 3z^2
func NewtonTwo(inColor bool) NewtonFunc {
	f := func(x complex128) complex128 {
		return x*x*x - 1
	}
	d := func (x complex128) complex128 {
		return 3*x*x
	}
	a := complex(1.0, 0)
	if inColor {
		return func(z complex128)  color.Color {
			return newtonColor(z, a, f, d)
		}
	}
	return func(z complex128) color.Color {
		return newtonMonochrome(z, a, f, d)
	}
}

// f(z) = 5cos(3z)
// f'(z) = -15sin(3z)
func NewtonThree(inColor bool) NewtonFunc {
	f := func(x complex128) complex128 {
		return 5*cmplx.Cos(3*x)
	}
	d := func (x complex128) complex128 {
		return -15*cmplx.Sin(3*x)
	}
	a := complex(1.0, 0)
	if inColor {
		return func(z complex128)  color.Color {
			return newtonColor(z, a, f, d)
		}
	}
	return func(z complex128) color.Color {
		return newtonMonochrome(z, a, f, d)
	}
}

// f(z) = ln(x)
// f'(z) = 1/x
func NewtonFour(inColor bool) NewtonFunc {
	f := func(x complex128) complex128 {
		return cmplx.Log(x)
	}
	d := func (x complex128) complex128 {
		return 1/x
	}
	a := complex(1.0, 0)
	if inColor {
		return func(z complex128)  color.Color {
			return newtonColor(z, a, f, d)
		}
	}
	return func(z complex128) color.Color {
		return newtonMonochrome(z, a, f, d)
	}
}

// f(z) = z^3 - 1
// f'(z) = 3z^2
// a = 2
func NewtonFive(inColor bool) NewtonFunc {
	f := func(x complex128) complex128 {
		return x*x*x - 1
	}
	d := func (x complex128) complex128 {
		return 3*x*x
	}
	a := complex(2, 0)
	if inColor {
		return func(z complex128)  color.Color {
			return newtonColor(z, a, f, d)
		}
	}
	return func(z complex128) color.Color {
		return newtonMonochrome(z, a, f, d)
	}
}

// f(z) = cosh(z) - 1
// f'(z) = sinh(z)
func NewtonSix(inColor bool) NewtonFunc {
	f := func(x complex128) complex128 {
		return cmplx.Cosh(x) - 1
	}
	d := func (x complex128) complex128 {
		return cmplx.Sinh(x)
	}
	a := complex(1, 0)
	if inColor {
		return func(z complex128)  color.Color {
			return newtonColor(z, a, f, d)
		}
	}
	return func(z complex128) color.Color {
		return newtonMonochrome(z, a, f, d)
	}
}
