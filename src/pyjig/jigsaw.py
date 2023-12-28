import base64
from math import ceil
import random
import subprocess
import numpy as np
import cairosvg
from PIL import Image
import defusedxml.ElementTree as ET
from svgpathtools import svg2paths
import tempfile
import os

"""
Generate jigsaw motifs and digital puzzle sets.

Functions:
generate_motif
generate_masks
generate_jigsaw
jigsaw_factory
"""


class Cut():
    def __init__(self, pieces_height, pieces_width, abs_height=None, abs_width=None, image=None, use_image=False, stroke_color="black", fill_color="white"):
        self.pieces_height = pieces_height
        self.pieces_width = pieces_width
        self.abs_height = abs_height
        self.abs_width = abs_width
        self.image = image
        self.stroke_color = stroke_color
        self.fill_color = fill_color
        self.use_image = use_image
        if use_image and self.image == None:
            raise Exception("No image provided")

        if (self.abs_height == None or self.abs_width == None) and self.image == None:
            raise Exception(
                "Height and width of the desired template must either be provided manually, or an image must be provided for it to be derived from.")

        if not use_image and (self.abs_height == None or self.abs_width == None):
            raise Exception(
                "Please either set a height and width or pass use_image=True in your function call")
        if use_image and self.image != None:
            width, height = Image.open(self.image).size
            self.abs_width = width
            self.abs_height = height

        print(self.update_cut_template())

    def update_cut_template(self):
        piece_w = self.abs_width // self.pieces_width
        piece_h = self.abs_height // self.pieces_height
        number_of_pieces = self.pieces_height * self.pieces_width
        col = 0
        paths = []
        all_commands = {}
        metadata = {
            "PiecesCount": number_of_pieces,
            "Rows": self.pieces_height,
            "Cols": self.pieces_width,
            "TotalWidth": self.abs_width,
            "TotalHeight": self.abs_height,
            "PieceWidth": piece_w,
            "PieceHeight": piece_h,
            "Pieces": []
        }

        # Create svg path for each piece
        for i in range(1, number_of_pieces+1):
            # Find grid position
            row = ceil(i / self.pieces_width)
            col = col + 1 if col < self.pieces_width else 1

            metadata["Pieces"].append({
                "PieceNumber": i,
                "UpperEdge": True if row == 1 else False,
                "LowerEdge": True if row == self.pieces_height else False,
                "LeftEdge": True if col == 1 else False,
                "RightEdge": True if col == self.pieces_width else False,
            })

            # Set pixel start and end positions
            origin_w = (col - 1) * piece_w
            origin_h = (row - 1) * piece_h
            x = origin_w + piece_w
            y = origin_h + piece_h

            # Calculate distance to the start of the notch
            to_v_notch = piece_h * 0.4
            to_h_notch = piece_w * 0.4
            v_notch = piece_h - (2 * to_v_notch)
            h_notch = piece_w - (2 * to_h_notch)

            # Control points for the puzzle notch curve, randomise direction
            curve_multiplier_1 = random.choice([0.85, 1.15])
            curve_multiplier_2 = 0.85 if curve_multiplier_1 != 0.85 else 1.15

            # Start command dictionary for storing commands for reuse on adjacent Pieces
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
            if col < self.pieces_width:
                # Generate curve
                r = "C {x},{origin_h} {w_curve_1},{half_piece_h} {x}, {to_notch_start} S {w_curve_2},{control_point} {x},{to_notch_end} S {x},{y} {x},{y}".format(
                    x=str(x),
                    y=str(y),
                    origin_h=origin_h,
                    origin_w=origin_w,
                    half_piece_h=str(origin_h+(piece_h * 0.5)),
                    w_curve_1=str(
                        origin_w + (piece_w * curve_multiplier_1)),
                    w_curve_2=str(
                        origin_w + (piece_w * curve_multiplier_2)),
                    to_notch_start=str(origin_h+to_v_notch),
                    to_notch_end=str(origin_h+to_v_notch+v_notch),
                    control_point=str(
                        origin_h+(piece_h * 0.5)+((to_v_notch+v_notch)-(piece_h * 0.5))*2)
                )

                # Create an inverted verseion for replicating the Left side of the adjacent piece
                r_inverted = "C {x},{y} {w_curve_1},{half_piece_h} {x}, {to_notch_start} S {w_curve_2},{control_point} {x},{to_notch_end} S {x},{origin_h} {x},{origin_h}".format(
                    x=str(x),
                    y=str(y),
                    origin_h=origin_h,
                    origin_w=origin_w,
                    half_piece_h=str(origin_h+(piece_h * 0.5)),
                    w_curve_1=str(
                        origin_w + (piece_w * curve_multiplier_1)),
                    w_curve_2=str(
                        origin_w + (piece_w * curve_multiplier_2)),
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
            if row < self.pieces_height:
                # Generate curve
                b = "C {x},{y} {half_piece_w},{w_curve_1} {to_notch_start},{y} S {control_point},{w_curve_2} {to_notch_end},{y} S {origin_w},{y} {origin_w},{y}".format(
                    x=str(x),
                    y=str(y),
                    origin_w=origin_w,
                    half_piece_w=str(origin_w+(piece_w * 0.5)),
                    w_curve_1=str(
                        origin_h + (piece_h * curve_multiplier_1)),
                    w_curve_2=str(
                        origin_h + (piece_h * curve_multiplier_2)),
                    to_notch_start=str(origin_w+to_h_notch+h_notch),
                    to_notch_end=str(origin_w+to_h_notch),
                    control_point=str(
                        origin_w+(piece_w * 0.5)-((to_h_notch+h_notch)-(piece_w * 0.5))*2)
                )

                # Create an inverted version for replicating the Left side of the adjacent piece
                b_inverted = "C {origin_w},{y} {half_piece_w},{w_curve_1} {to_notch_start},{y} S {control_point},{w_curve_2} {to_notch_end},{y} S {x},{y} {x},{y}".format(
                    x=str(x),
                    y=str(y),
                    origin_h=origin_h,
                    origin_w=origin_w,
                    half_piece_w=str(origin_w+(piece_w * 0.5)),
                    w_curve_1=str(
                        origin_h + (piece_h * curve_multiplier_1)),
                    w_curve_2=str(
                        origin_h + (piece_h * curve_multiplier_2)),
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

            # Close path (including straight line if Left edge piece)
            commands.append("z")

            # Construct path element
            d = "\n\t".join(commands)
            path = '<path stroke="{}" fill="{}" d="{}" />'.format(
                self.stroke_color, self.fill_color, d)
            paths.append(path)

        paths = "\n\t".join(paths)
        svg_template = """\
    <svg width="{}" height="{}">
        {}
    </svg>
        """.format(self.abs_width, self.abs_height, paths)

        self.svg_template = svg_template

        self.metadata = metadata

        return "Puzzle template update complete"

    def to_svg(self, filepath):
        print(self.update_cut_template())
        svg_file = open(filepath, "w")
        svg_file.write(self.svg_template)
        svg_file.close()

        return "Puzzle cut template created {}".format(filepath)


# mycut = Cut(5, 4, image="./Zugpsitze_mountain.jpg", use_image=True)


def image_encode(original_image):
    ext = original_image.split(".")[1]
    ext = "jpeg" if ext == "jpg" else ext
    with open(original_image, "rb") as image:
        encoded_string = base64.b64encode(image.read()).decode('utf-8')
    return (ext, encoded_string)


class Jigsaw():
    def __init__(self, cut: Cut, image=None):
        self.cut = cut
        self.image = image

    def generate_svg_jigsaw(self, outdirectory):
        metadata = self.cut.metadata
        fp = tempfile.NamedTemporaryFile(suffix=".SVG")

        fp.write(self.cut.svg_template.encode('utf-8'))

        bboxes = subprocess.check_output(
            ["inkscape", "--query-all", "{}".format(fp.name)]).decode('utf-8')
        bboxes = bboxes.split("\n")[1:-1]
        ext, encoded = image_encode(self.image)
        paths, _ = svg2paths(fp.name)
        fp.close()

        # Apply bounding box for each path and generate svg from template
        for p, path in enumerate(paths):
            xmin, ymin, width, height = bboxes[p].split(",")[1:]

            top_left_corner = True if metadata["Pieces"][p]["UpperEdge"] and metadata["Pieces"][p]["LeftEdge"] else False
            top_right_corner = True if metadata["Pieces"][p][
                "UpperEdge"] and metadata["Pieces"][p]["RightEdge"] else False
            bottom_left_corner = True if metadata["Pieces"][p][
                "LowerEdge"] and metadata["Pieces"][p]["LeftEdge"] else False
            bottom_right_corner = True if metadata["Pieces"][p][
                "LowerEdge"] and metadata["Pieces"][p]["RightEdge"] else False

            top_anchor = 1
            if top_left_corner or metadata["Pieces"][p]["UpperEdge"]:
                right_anchor = 3
                left_anchor = 27
            elif top_right_corner:
                right_anchor = 3
                left_anchor = 17
            elif metadata["Pieces"][p]["RightEdge"]:
                right_anchor = 13
                left_anchor = 27
            elif bottom_right_corner:
                right_anchor = 13
                left_anchor = 17
            elif bottom_left_corner or metadata["Pieces"][p]["LowerEdge"]:
                right_anchor = 13
                left_anchor = 27
            else:
                right_anchor = 13
                left_anchor = 37

            midpoint_top = (float(path.d().split(" ")[top_anchor].split(
                ",")[0]) - float(xmin)) + (metadata["PieceWidth"] / 2)
            midpoint_right = (float(path.d().split(" ")[right_anchor].split(
                ",")[1]) - float(ymin)) + (metadata["PieceHeight"] / 2)
            midpoint_bottom = (float(path.d().split(" ")[left_anchor].split(
                ",")[0]) - float(xmin)) + (metadata["PieceWidth"] / 2)
            midpoint_left = (float(path.d().split(" ")[top_anchor].split(
                ",")[1]) - float(ymin)) + (metadata["PieceHeight"] / 2)

            metadata["Pieces"][p]["MidpointTop"] = midpoint_top
            metadata["Pieces"][p]["MidpointRight"] = midpoint_right
            metadata["Pieces"][p]["MidpointBottom"] = midpoint_bottom
            metadata["Pieces"][p]["MidpointLeft"] = midpoint_left

            svg = """\
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="{} {} {w} {h}" width="{w}" height="{h}">
        <defs>
            <path id="cropPath" d="{d}" />
            <clipPath id="crop">
                <use href="#cropPath" />
            </clipPath>
        </defs>
        <image href="data:image/{ext};base64,{encoded}" clip-path="url(#crop)"/>
    </svg>
    """.format(xmin, ymin, w=width, h=height, d=path.d(), ext=ext, encoded=encoded)
            with open(os.path.join(outdirectory, '{}.svg'.format(p)), 'w') as file:
                file.write(svg)

        self.cut.metadata = metadata

        return "Svg puzzle set generated: {} ({} Pieces) Directory: {}".format(self.image, len(paths), outdirectory)


# myjig = Jigsaw(mycut, "Zugpsitze_mountain.jpg")
# print(myjig.generate_svg_jigsaw("./Pieces"))


def jigsaw_factory():
    """Execute the motif, masks and jigsaw logic sequentially"""
    pass
