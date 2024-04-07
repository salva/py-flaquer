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

def main():

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

    args = ap.parse_args()

    img = cv2.imread(args.input)
    (img_h, img_w, _) = img.shape

    columns = args.columns
    rows = int(columns * img_h / img_w + 0.5)

    cell = min(img_h // rows, img_w // columns)
    offset_x = (img_w - columns * cell) // 2
    offset_y = (img_h - rows * cell) // 2

    output_cell = int(args.width / columns + 0.5)

    width = output_cell * columns
    height = output_cell * rows

    background = np.array(color_by_name[args.background])
    background_opposite = np.array(opposite_color(background))

    # create output image repeating background color tuple
    output = np.full((height, width, 3), background, dtype=np.uint8)

    degradations = args.degradations
    p = np.array([background] + palette[args.palette])

    ps = [p, [background_opposite]]
    if degradations > 0:
        for d in np.linspace(1.0, 0.0, degradations + 2, endpoint=False):
            ps.append(p * d + background_opposite * (1 - d))
    colors = np.concatenate(ps)
    area_reduction = args.area_reduction

    for i in range(rows):
        for j in range(columns):
            x = offset_x + j * cell
            y = offset_y + i * cell

            color = img[y:y+cell, x:x+cell].mean(axis=(0, 1)).astype(np.uint8)

            ### A = np.concatenate(colors.T, [np.ones(colors.shape[0])])
            A = np.concatenate([colors.T, np.ones((1, colors.shape[0]))], axis=0)
            b = np.array([*color, 1.0])

            x, _ = nnls(A, b)
            # print(f"x.shape = {x.shape}, colors.shape = {colors.shape}")

            pairs = list(zip(x, colors))
            pairs = pairs[1:] # remove background!
            pairs = sorted(pairs, key=lambda x: x[0])

            acu = 0
            triplets = []
            for x, c in pairs:
                if x > 0:
                    acu += x
                    triplets.append((acu, x, c))

            for area, _, c in triplets[::-1]:
                center = (j * output_cell + output_cell // 2, i * output_cell + output_cell // 2)
                r = int(math.sqrt(area * output_cell * output_cell / math.pi * area_reduction) + 0.5)
                # print(f"({i}, {j}) -> {area} -> {r}, color = {c}, output.shape = {output.shape}")


                cv2.circle(output, center, r, c.tolist(), -1)


    cv2.imwrite(args.output, output)

if __name__ == '__main__':
    main()

