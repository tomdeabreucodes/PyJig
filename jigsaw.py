import io
from math import ceil
import random
import cv2
import numpy as np
from decorators import timer
import cairosvg
import io
from PIL import Image
import defusedxml.ElementTree as ET

"""
Generate jigsaw motifs and digital puzzle sets.

Functions:
generate_motif
generate_masks
generate_jigsaw
jigsaw_factory
"""


@timer
def generate_motif(pieces_height, pieces_width, abs_height=100, abs_width=100):
    """Generate a jigsaw motif (template) to be used as the cutting lines"""
    number_of_pieces = pieces_height * pieces_width
    col = 0
    paths = []
    all_commands = {}

    # Create svg path for each piece
    for i in range(1, number_of_pieces+1):
        # Find grid position
        row = ceil(i / pieces_width)
        col = col + 1 if col < pieces_width else 1

        # Calculate pixel dimentions of each piece
        piece_w = abs_width // pieces_width
        piece_h = abs_height // pieces_height

        # Set pixel start and end positions
        origin_w = (col - 1) * piece_w
        origin_h = (row - 1) * piece_h
        x = origin_w + piece_w
        y = origin_h + piece_h

        # Calculate distance to the start of the
        to_v_notch = piece_h * 0.4
        to_h_notch = piece_w * 0.4
        v_notch = piece_h - (2 * to_v_notch)
        h_notch = piece_w - (2 * to_h_notch)

        # Control points for the puzzle notch curve, randomise direction
        curve_multiplier_1 = random.choice([0.85, 1.15])
        curve_multiplier_2 = 0.85 if curve_multiplier_1 != 0.85 else 1.15

        # Start command dictionary for storing commands for reuse on adjacent pieces
        commands = []
        commands.append("M {}, {}".format(origin_w, origin_h))

        # Top section
        if row > 1:
            # Use inverted command from adjacent piece
            t = all_commands['{}-{}-t'.format(row, col)]
        else:
            # Edge piece
            t = "L {}, {}".format(str(x), str(origin_h))
        commands.append(t)
        all_commands["{}-{}-t".format(row, col)] = t

        # Right section
        if col < pieces_width:
            # Generate curve
            r = "C {x},{origin_h} {w_curve_1},{half_piece_h} {x}, {to_notch_start} S {w_curve_2},{control_point} {x},{to_notch_end} S {x},{y} {x},{y}".format(
                x=str(x),
                y=str(y),
                origin_h=origin_h,
                origin_w=origin_w,
                half_piece_h=str(origin_h+(piece_h * 0.5)),
                w_curve_1=str(origin_w + (piece_w * curve_multiplier_1)),
                w_curve_2=str(origin_w + (piece_w * curve_multiplier_2)),
                to_notch_start=str(origin_h+to_v_notch),
                to_notch_end=str(origin_h+to_v_notch+v_notch),
                control_point=str(
                    origin_h+(piece_h * 0.5)+((to_v_notch+v_notch)-(piece_h * 0.5))*2)
            )

            # Create an inverted verseion for replicating the left side of the adjacent piece
            r_inverted = "C {x},{y} {w_curve_1},{half_piece_h} {x}, {to_notch_start} S {w_curve_2},{control_point} {x},{to_notch_end} S {x},{origin_h} {x},{origin_h}".format(
                x=str(x),
                y=str(y),
                origin_h=origin_h,
                origin_w=origin_w,
                half_piece_h=str(origin_h+(piece_h * 0.5)),
                w_curve_1=str(origin_w + (piece_w * curve_multiplier_1)),
                w_curve_2=str(origin_w + (piece_w * curve_multiplier_2)),
                to_notch_start=str(origin_h+to_v_notch+v_notch),
                to_notch_end=str(origin_h+to_v_notch),
                control_point=str(
                    origin_h+(piece_h * 0.5)-((to_v_notch+v_notch)-(piece_h * 0.5))*2)
            )
            all_commands["{}-{}-l".format(row, col + 1)] = r_inverted
        else:
            # Edge piece
            r = "L {},{}".format(x, y)

        commands.append(r)

        # Bottom section
        if row < pieces_height:
            # Generate curve
            b = "C {x},{y} {half_piece_w},{w_curve_1} {to_notch_start},{y} S {control_point},{w_curve_2} {to_notch_end},{y} S {origin_w},{y} {origin_w},{y}".format(
                x=str(x),
                y=str(y),
                origin_h=origin_h,
                origin_w=origin_w,
                half_piece_w=str(origin_w+(piece_w * 0.5)),
                w_curve_1=str(origin_h + (piece_h * curve_multiplier_1)),
                w_curve_2=str(origin_h + (piece_h * curve_multiplier_2)),
                to_notch_start=str(origin_w+to_h_notch+h_notch),
                to_notch_end=str(origin_w+to_h_notch),
                control_point=str(
                    origin_w+(piece_w * 0.5)-((to_h_notch+h_notch)-(piece_w * 0.5))*2)
            )

            # Create an inverted verseion for replicating the left side of the adjacent piece
            b_inverted = "C {origin_w},{y} {half_piece_w},{w_curve_1} {to_notch_start},{y} S {control_point},{w_curve_2} {to_notch_end},{y} S {x},{y} {x},{y}".format(
                x=str(x),
                y=str(y),
                origin_h=origin_h,
                origin_w=origin_w,
                half_piece_w=str(origin_w+(piece_w * 0.5)),
                w_curve_1=str(origin_h + (piece_h * curve_multiplier_1)),
                w_curve_2=str(origin_h + (piece_h * curve_multiplier_2)),
                to_notch_start=str(origin_w+to_h_notch),
                to_notch_end=str(origin_w+to_h_notch+h_notch),
                control_point=str(
                    origin_w+(piece_w * 0.5)+((to_h_notch+h_notch)-(piece_w * 0.5))*2)
            )
            all_commands["{}-{}-t".format(row + 1, col)] = b_inverted
        else:
            # Edge piece
            b = "L {},{}".format(origin_w, y)

        commands.append(b)

        if col > 1:
            l = all_commands["{}-{}-l".format(row, col)]
            commands.append(l)

        # Close path (including straight line if left edge piece)
        commands.append("z")

        # Construct path element
        d = "\n\t".join(commands)
        path = '<path stroke="white" fill="white" d="{}" />'.format(d)
        paths.append(path)

    paths = "\n\t".join(paths)
    svg_template = """\
<svg width="{}" height="{}">
    {}
</svg>
    """.format(abs_width, abs_height, paths)
    svg_file = open("motif.svg", "w")
    svg_file.write(svg_template)
    svg_file.close()


generate_motif(5, 8, 630, 1200)


@timer
def generate_masks(motif_file):
    """Generate binary masks from the image and motif"""

    # Get each path from motif file
    tree = ET.parse(motif_file)
    root = tree.getroot()
    width, height = root.attrib["width"], tree.getroot().attrib["height"]
    paths = root.findall('path')

    # Create a binary mask per jigsaw piece
    masks = []
    for p, path in enumerate(paths):
        # Create a new SVG file with just the current path element
        new_svg = f'<svg shape-rendering="crispEdges" width="{width}" height="{height}">{ET.tostring(path)}</svg>'

        # Convert the SVG file to a PNG image using cairosvg
        mem = io.BytesIO()
        cairosvg.svg2png(bytestring=new_svg,
                         write_to=mem, background_color=None,)
        masks.append(np.array(Image.open(mem)))
    return masks


masks = generate_masks('motif.svg')


@timer
def generate_jigsaw(original_image, masks):
    """Generate set of puzzle pieces as individual .PNG files"""
    image = cv2.imread(original_image)
    for m, mask in enumerate(masks):
        # Apply mask
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        result = cv2.bitwise_and(image, image, mask=mask)

        # Crop to match minimum size (bbox)
        contours = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        cntr = contours[0]
        x, y, w, h = cv2.boundingRect(cntr)

        # Add buffer to avoid harsh cutoffs when softening edges
        if x > 0:
            x -= 4
            w += 8
        else:
            w += 4
        if y > 0:
            y -= 4
            h += 8
        else:
            h += 4

        # Soften edges
        edge_mask = np.zeros(image.shape, dtype=np.uint8)
        cv2.drawContours(edge_mask, contours, -1, (255, 255, 255), 3)
        blur_piece = cv2.GaussianBlur(result, (9, 9), 0)
        final_piece = np.where(edge_mask == np.array(
            [255, 255, 255]), blur_piece, result)

        # Add transparency and save piece as PNG
        final_piece = cv2.cvtColor(final_piece, cv2.COLOR_BGR2BGRA)
        final_piece[:, :, 3] = mask
        final_piece = final_piece[y:y+h, x:x+w]
        cv2.imwrite(f"./pieces/{m}.png", final_piece)


generate_jigsaw("Zugpsitze_mountain.jpg", masks)


def jigsaw_factory():
    """Execute the motif, masks and jigsaw logic sequentially"""
    pass
