This Python package provides functionality to generate jigsaw puzzle cut templates and digital puzzle sets. It includes functions for creating jigsaw templates, applying cuts, and generating SVG representations.
## Installation

You can install the package using pip:

```shell
pip install pyjigsaw
```

## Usage

```python
from pyjigsaw import jigsawfactory

# Create a jigsaw cut template
mycut = jigsawfactory.Cut(5, 4, image="./Zugpsitze_mountain.jpg", use_image=True)

# Generate a jigsaw set from the cut template
myjig = jigsawfactory.Jigsaw(mycut, "Zugpsitze_mountain.jpg")
myjig.generate_svg_jigsaw("./Pieces")
```
## Features
### Cut Class

The Cut class is responsible for creating jigsaw cut templates. It takes parameters such as the number of pieces in height and width, absolute height and width, an optional image for deriving dimensions, and stroke/fill color specifications.

```python
from pyjigsaw import jigsawfactory


# Create a jigsaw cut template
mycut = jigsawfactory.Cut(5, 4, image="./Zugpsitze_mountain.jpg", use_image=True)
```

### Jigsaw Class

The Jigsaw class uses a Cut template to generate a set of jigsaw pieces in SVG format. It takes a Cut object and an image file path. The generated SVG files are saved in the specified output directory.
```python
from pyjigsaw import jigsawfactory

# Create a jigsaw set from the cut template
myjig = jigsawfactory.Jigsaw(mycut, "Zugpsitze_mountain.jpg")
myjig.generate_svg_jigsaw("./Pieces")
```

## Contributing

Feel free to contribute to the project by opening issues or submitting pull requests on GitHub.
License

This project is licensed under the MIT License - see the LICENSE file for details.