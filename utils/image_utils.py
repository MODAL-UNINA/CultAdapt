from PIL import Image


def composite_to_rgb(img):
    if img.mode in ('RGBA', 'LA', 'P'):
        rgba = img.convert('RGBA')
        background = Image.new('RGBA', rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(background, rgba).convert('RGB')
    return img.convert('RGB')


def safe_open_image(image_path, verbose=False):
    try:
        with Image.open(image_path) as img:
            img.load()
            return composite_to_rgb(img.copy())
    except Exception as exc:
        raise ValueError(f'Cannot read image {image_path!r}: {exc}') from exc
