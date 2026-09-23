from PIL import Image, ImageDraw, ImageFont
import arabic_reshaper
from bidi.algorithm import get_display

text = "سب کافر دشمنوں کا سر تن سے جدا کر دو"
text = get_display(arabic_reshaper.reshape(text))

font = ImageFont.truetype(
    "src/urdu_lime/NotoNastaliqUrdu-Regular.ttf",
    48,
    layout_engine=ImageFont.LAYOUT_RAQM
)

img = Image.new("RGB", (1400, 250), "white")
draw = ImageDraw.Draw(img)

draw.text(
    (1350, 120),
    text,
    fill="black",
    font=font,
    anchor="ra"
)

img.show()
