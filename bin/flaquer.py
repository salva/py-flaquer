import sys
import argparse
import cv2
import numpy as np
from scipy.optimize import nnls
import math

palette = { 'primary': [[0, 0, 255], [0, 255, 0], [255, 0, 0]],
            'secondary': [[255, 255, 0], [255, 0, 255], [0, 255, 255]],
            'primary-secondary': [[0, 0, 255], [0, 255, 0], [255, 0, 0],
                                  [255, 255, 0], [255, 0, 255], [0, 255, 255]],
            'wheel': [[255, 0, 0], [255, 192, 0], [255, 255, 0], [192, 256, 0], [0, 255, 0], [0, 255, 192], [0, 255, 255], [0, 192, 255], [0, 0, 255], [192, 0, 255], [255, 0, 255], [255, 0, 192]] }
color_by_name = { 'black': [0, 0, 0],
                  'white': [255, 255, 255] }

class Drawer:

    def __init__(self, args, output_cell_area, cols, rows):
        self._output_cell_area = output_cell_area
        self._cols = cols
        self._rows = rows

        ap = argparse.ArgumentParser()
        self._add_arguments(ap)

        self._args = ap.parse_args(args)

    def _add_arguments(self, ap):
        pass

    def pairs_to_triplets(self, pairs):
        acu = 0
        triplets = []
        for v, c in sorted(pairs, key=lambda p: p[0]):
            if v > 0:
                acu += v
                triplets.append((acu, v, c))
        return triplets

class CircleDrawer(Drawer):

    def draw(self, img, pos, center, pairs):
        triplets = self.pairs_to_triplets(pairs)

        for ratio, _, c in triplets[::-1]:
            r = int(math.sqrt(ratio * self._output_cell_area / math.pi) + 0.5)
            cv2.circle(img, center, r, c.tolist(), -1)


font_by_name = {
    "simplex": cv2.FONT_HERSHEY_SIMPLEX,
    "plain": cv2.FONT_HERSHEY_PLAIN,
    "duplex": cv2.FONT_HERSHEY_DUPLEX,
    "complex": cv2.FONT_HERSHEY_COMPLEX,
    "triplex": cv2.FONT_HERSHEY_TRIPLEX,
    "complex_small": cv2.FONT_HERSHEY_COMPLEX_SMALL,
    "script_simplex": cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
    "script_complex": cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
    "complex_sc": cv2.FONT_HERSHEY_COMPLEX_SC,
    "triplex_sc": cv2.FONT_HERSHEY_TRIPLEX_SC,
}

class TextDrawer(Drawer):
    def _add_arguments(self, ap):
        ap.add_argument('-f', '--font', help='Font to use', default='simplex')
        ap.add_argument('-m', '--min-rel-size', help='Minimum font size', default=0.3, type=float)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._font = font_by_name[self._args.font]
        self._measure_font()
        
    def _measure_font(self):

        self._font_size_cache = {}
        
        diameter = int(sqrt(self._output_cell_area) + 0.5)
        test_img = np.zeros((diameter*3, diameter*3), dtype=np.uint8)

        for scale in range(1, 100):
            size = cv2.getTextSize('A', self._font, scale, 1)[0]

            if size > 0.6 * diameter:
                break

            for c in range(64, 256):
                char = chr(c)

                for thickness in range(1, 100):
                    test_img.fill(0)
                    cv2.putText(test_img, char, (diameter, 2*diameter), self._font, scale, 1, thickness)

                    area = cv2.countNonZero(test_img)
                    ratio = area / self._output_cell_area


        self._text_size = {scale: cv2.getTextSize('A', self._font, scale, 1)[0] for scale in range(1, 100)}

        size = int(sqrt(self._output_cell_area)) * 3 + 2

        
        for c in range(64, 256):
            char = chr(c)
            
        
    def draw(self, img, pos, center, pairs):
        triplets = self.pairs_to_triplets(pairs)

        for ratio, _, c in triplets[::-1]:
            cv2.putText(img, chr(65 + pos[1] % 26), center, self._font, 5.0, c.tolist(), int(100 * ratio))

drawer_class_by_name = { 'circle': CircleDrawer,
                         'text': TextDrawer }

def opposite_color(color):
    return [0 if c > 127 else 255 for c in color]

class BasePattern:

    def __init__(self, args):

        ap = argparse.ArgumentParser()
        ap.add_argument('-o', '--output', help='Output file', default='output.png')
        ap.add_argument('-w', '--width', help='Width of the output image', default=4096, type=int)
        ap.add_argument('-c', '--columns', help='Number of columns', default=80, type=int)
        ap.add_argument('-b', '--background', help='Background color', default='white')
        ap.add_argument('-P', '--palette', help='Palette to use', default='primary')
        ap.add_argument('-g', '--degradations', help='Number of degradations', default=0, type=int)
        ap.add_argument('-r', '--area-reduction', help='Area reduction factor', default=0.85, type=float)
        ap.add_argument('-d', '--drawer', help='Drawer to use', type=str, default='circle')

        ap.add_argument('input', help='Input file')

        self.add_more_arguments(ap)

        (self._args, self._drawer_args) = ap.parse_known_args(args)

    def add_more_arguments(self, ap):
        pass

    def run(self):
        self.init_input()
        self.init_colors()
        self.init_dimensions()
        self.init_output()
        self.init_drawer()
        self.loop()
        self.save_output()

    def init_input(self):
        self._img = cv2.imread(self._args.input)
        (self._input_h, self._input_w, _) = self._img.shape

    def init_colors(self):
        self._background = (bg := np.array(color_by_name[self._args.background]))
        bg_opposite = np.array(opposite_color(bg))

        degradations = self._args.degradations
        p = np.array([bg] + palette[self._args.palette])

        ps = [p, [bg_opposite]]
        if degradations > 0:
            for d in np.linspace(1.0, 0.0, degradations + 2, endpoint=False):
                ps.append(p * d + bg_opposite * (1 - d))
        self._colors = np.concatenate(ps)

    def init_drawer(self):
        drawer_class = drawer_class_by_name[self._args.drawer]
        self._drawer = drawer_class(self._drawer_args,
                                    output_cell_area=self._output_cell_area * self._args.area_reduction,
                                    cols=self._cols,
                                    rows=self._rows)

    def process_cell(self, i, j):
        color = self.input_cell_color(i, j)

        A = np.concatenate([self._colors.T, np.ones((1, self._colors.shape[0]))], axis=0)
        b = np.array([*color, 1.0])

        vs, _ = nnls(A, b)

        pairs = list(zip(vs, self._colors))
        pairs = pairs[1:]

        center = self.output_cell_center(i, j)

        self._drawer.draw(self._output, (i, j), center, pairs)


    def save_output(self):
        cv2.imwrite(self._args.output, self._output)


class SquarePattern(BasePattern):

    def init_dimensions(self):
        self._cols = (cols := self._args.columns)
        self._input_cell = (input_cell := self._input_w // cols)
        self._rows = (rows := int(self._input_h / input_cell))
        self._input_offset_x = (self._input_w - cols * input_cell) // 2
        self._input_offset_y = (self._input_h - rows * input_cell) // 2

    def init_output(self):
        self._output_cell = (output_cell := int(self._args.width / self._cols + 0.5))
        self._output_cell_area = output_cell * output_cell
        self._output_width = (width := output_cell * self._cols)
        self._output_height = (height := output_cell * self._rows)
        self._output = np.full((height, width, 3), self._background, dtype=np.uint8)

    def loop(self):
        for i in range(self._rows):
            for j in range(self._cols):
                self.process_cell(i, j)

    def input_cell_color(self, i, j):
        x = self._input_offset_x + j * self._input_cell
        y = self._input_offset_y + i * self._input_cell
        return self._img[y:y+self._input_cell, x:x+self._input_cell].mean(axis=(0, 1)).astype(np.uint8)

    def output_cell_center(self, i, j):
        x = j * self._output_cell + self._output_cell // 2
        y = i * self._output_cell + self._output_cell // 2
        return (x, y)


class HexagonalPattern(BasePattern):

    def init_dimensions(self):

        self._cols = (cols := self._args.columns)
        self._input_cell_half_width = (hw := (self._input_w / cols / 2))
        self._input_cell_radius = (r := 2 / math.sqrt(3) * hw)

        self._rows = (rows := int((self._input_h / r - 0.5) / 1.5))

        self._input_offset_x0 = self._input_w / 2 - hw * cols
        self._input_offset_x1 = self._input_w / 2 - hw * (cols - 1)
        self._input_offset_y = (self._input_h - r * (rows * 1.5 + 0.5)) * 0.5

    def init_output(self):

        self._output_cell_half_width = (hw := int(self._args.width / self._cols / 2 + 0.5))
        self._output_cell_radius = (r := 2 * hw / math.sqrt(3))
        self._output_cell_area = hw * r * 3
        self._output_cell_quarter_height = (qh := r * 0.5 + 0.5)
        self._output_width = (width := hw * self._cols * 2)
        self._output_height = (height := int(qh * (self._rows * 3 + 1) + 0.5))
        self._output = np.full((height, width, 3), self._background, dtype=np.uint8)

    def loop(self):
        for i in range(self._rows):
            cols = self._cols if i % 2 == 0 else self._cols - 1
            for j in range(cols):
                self.process_cell(i, j)

    def input_cell_color(self, i, j):
        offset_x = self._input_offset_x0 if i % 2 == 0 else self._input_offset_x1
        x0 = int(offset_x + j * self._input_cell_half_width * 2)
        x1 = int(offset_x + (j + 1) * self._input_cell_half_width * 2)

        y0 = int(self._input_offset_y + (i * 1.5 + 0.25) * self._input_cell_radius)
        y1 = int(self._input_offset_y + (i * 1.5 + 1.75) * self._input_cell_radius)
        return self._img[y0:y1, x0:x1].mean(axis=(0, 1)).astype(np.uint8)

    def output_cell_center(self, i, j):
        offset_x = 0 if i % 2 == 0 else self._output_cell_half_width
        x = int(offset_x + (2 * j + 1) * self._output_cell_half_width + 0.5)
        y = int((3 * i + 2) * self._output_cell_quarter_height + 0.5)
        return (x, y)


pattern_class_by_name = { 'square': SquarePattern,
                          'hexagonal': HexagonalPattern }

def main():

    ap = argparse.ArgumentParser()
    ap.add_argument('-p', '--pattern', help='Pattern to use', type=str, default='square')

    args, rest = ap.parse_known_args()
    pattern_class_by_name[args.pattern](rest).run()


if __name__ == '__main__':
    main()

