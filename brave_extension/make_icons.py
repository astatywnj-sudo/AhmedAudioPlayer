from PIL import Image, ImageDraw

def make_icon(size, filename):
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    margin = max(1, size // 16)
    draw.ellipse(
        [margin, margin, size - margin, size - margin],
        fill=(26, 20, 16, 255)
    )

    cx, cy = size // 2, size // 2
    bar_w = max(1, size // 16)
    bar_h = size // 3
    gap = max(2, size // 12)

    for i, h_scale in enumerate([0.4, 0.8, 1.0, 0.8, 0.4]):
        x = cx - (2 * gap) + (i * gap)
        h = int(bar_h * h_scale)
        draw.rounded_rectangle(
            [x - bar_w // 2, cy - h // 2, x + bar_w // 2, cy + h // 2],
            radius=bar_w // 2,
            fill=(232, 144, 64, 255)
        )

    img.save(filename, "PNG")
    print("Made:", filename)

make_icon(16, "icon16.png")
make_icon(48, "icon48.png")
make_icon(128, "icon128.png")