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

def opposite_color(color):
    return [0 if c > 127 else 255 for c in color]

class SquarePattern:

    def __init__(self, args):

        ap = argparse.ArgumentParser()
        ap.add_argument('-o', '--output', help='Output file', default='output.png')
        ap.add_argument('-w', '--width', help='Width of the output image', default=4096, type=int)
        ap.add_argument('-c', '--columns', help='Number of columns', default=80, type=int)
        ap.add_argument('-p', '--pattern', help='Pattern to use', default='square')
        ap.add_argument('-b', '--background', help='Background color', default='white')
        ap.add_argument('-P', '--palette', help='Palette to use', default='primary')
        ap.add_argument('-g', '--degradations', help='Number of degradations', default=0, type=int)
        ap.add_argument('-r', '--area-reduction', help='Area reduction factor', default=0.85, type=float)
        ap.add_argument('input', help='Input file')

        self.add_more_arguments(ap)

        self._args = ap.parse_args(args)

    def add_more_arguments(self, ap):
        pass

    def run(self):
        self.init_input()
        self.init_colors()
        self.init_dimensions()
        self.init_output()
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

    def init_dimensions(self):
        self._cols = (cols := self._args.columns)
        w = self._args.width
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


    def process_cell(self, i, j):
        color = self.input_cell_color(i, j)

        A = np.concatenate([self._colors.T, np.ones((1, self._colors.shape[0]))], axis=0)
        b = np.array([*color, 1.0])

        vs, _ = nnls(A, b)

        pairs = list(zip(vs, self._colors))
        pairs = pairs[1:]

        pairs = sorted(pairs, key=lambda p: p[0])

        acu = 0
        triplets = []
        for v, c in pairs:
            if v > 0:
                    acu += v
                    triplets.append((acu, v, c))

        center = self.output_cell_center(i, j)
        for ratio, _, c in triplets[::-1]:
            r = int(math.sqrt(ratio * self._output_cell_area / math.pi * self._args.area_reduction) + 0.5)
            cv2.circle(self._output, center, r, c.tolist(), -1)


    def input_cell_color(self, i, j):
        x = self._input_offset_x + j * self._input_cell
        y = self._input_offset_y + i * self._input_cell
        return self._img[y:y+self._input_cell, x:x+self._input_cell].mean(axis=(0, 1)).astype(np.uint8)

    def output_cell_center(self, i, j):
        x = j * self._output_cell + self._output_cell // 2
        y = i * self._output_cell + self._output_cell // 2
        return (x, y)

    def save_output(self):
        cv2.imwrite(self._args.output, self._output)

def main():
    SquarePattern(sys.argv[1:]).run()

if __name__ == '__main__':
    main()

