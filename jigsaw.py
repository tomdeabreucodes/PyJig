from math import ceil
import random

"""
Generate jigsaw motifs and digital puzzle sets.

Functions:
generate_motif
generate_masks
generate_jigsaw
jigsaw_factory
"""


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
            t = all_commands['{}-{}-t'.format(row, col)]
        else:
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

        # all_commands["{}-{}-b".format(row, col)] = b
        if col > 1:
            l = all_commands["{}-{}-l".format(row, col)]
            commands.append(l)
        # elif col == pieces_width:

        commands.append("z")

        d = "\n\t".join(commands)
        color = "#"+''.join([random.choice('0123456789ABCDEF')
                            for j in range(6)])
        path = '<path stroke="{}" fill="{}" d="{}" />'.format(color, color, d)
        paths.append(path)

    print(all_commands)
    # print(i, row, col)
    paths = "\n\t".join(paths)
    svg_template = """\
<svg width="{}" height="{}">
    {}
    <circle cx="200" cy="0" r="2"/>
    <circle cx="170" cy="106.5" r="2"/>
    <circle cx="230" cy="150" r="2"/>
    <circle cx="200" cy="213" r="2"/>
    <circle fill="red" cx="200" cy="85.2" r="2"/>
    <circle fill="red" cx="200" cy="127.8" r="2"/>
</svg>
    """.format(abs_width, abs_height, paths)
    svg_file = open("motif.svg", "w")
    svg_file.write(svg_template)
    svg_file.close()


generate_motif(5, 4, 1065, 800)


def generate_masks(motif_file, image_file):
    """Generate binary masks from the image and motif"""


def generate_jigsaw():
    """Generate set of puzzle pieces as individual .PNG files"""
    pass


def jigsaw_factory():
    """Execute the motif, masks and jigsaw logic sequentially"""
    pass
