import fitz
from PIL import Image, ImageDraw, ImageFont
import random, math, os
import cv2
import numpy as np


def extract_text_blocks(pdf_path):
    doc = fitz.open(pdf_path)
    all_blocks = []

    for page_num, page in enumerate(doc):
        blocks = page.get_text("blocks")
        for b in blocks:
            x0, y0, x1, y1, text, *_ = b
            if text.strip():
                all_blocks.append({
                    "page": page_num,
                    "bbox": (x0, y0, x1, y1),
                    "text": text.strip()
                })
    return all_blocks


def apply_pressure_variation(ch_img, pressure):
    """Simulate pen pressure by varying stroke thickness and darkness."""
    img_array = np.array(ch_img)
    alpha = img_array[:, :, 3]
    
    # Lighter pressure = thinner, lighter strokes
    if pressure < 0.7:
        kernel_size = 2
        alpha = cv2.erode(alpha, np.ones((kernel_size, kernel_size), np.uint8), iterations=1)
        # Lighten the color
        img_array[:, :, :3] = np.clip(img_array[:, :, :3] + (255 - img_array[:, :, :3]) * (1 - pressure) * 0.5, 0, 255).astype(np.uint8)
    # Heavier pressure = thicker, darker strokes
    elif pressure > 1.3:
        kernel_size = 2
        alpha = cv2.dilate(alpha, np.ones((kernel_size, kernel_size), np.uint8), iterations=1)
        # Darken the color
        img_array[:, :, :3] = (img_array[:, :, :3] * 0.85).astype(np.uint8)
    
    img_array[:, :, 3] = alpha
    return Image.fromarray(img_array)





def draw_handwritten_line(img, draw, text, x, y, font, ink_color):
    """Draw text with realistic human variations: pressure, kerning, shape variation."""
    cursor_x = x
    baseline_shift = random.uniform(-3, 3)  

    word_gap_base = 14        
    word_gap_jitter = 2
    
    # Pen pressure state (simulates hand pressure variation)
    pressure = random.uniform(0.9, 1.1)
    pressure_momentum = random.uniform(-0.05, 0.05)

    for ch in text:
        # If this is a space character, advance by the word gap and skip drawing
        if ch == " ":
            cursor_x += word_gap_base + random.uniform(-word_gap_jitter, word_gap_jitter)
            # Pen lift/reset pressure slightly
            pressure = random.uniform(0.9, 1.1)
            continue
            
        # Gradually evolving pressure (like human hand)
        pressure += pressure_momentum + random.uniform(-0.1, 0.1)
        pressure = np.clip(pressure, 0.5, 1.5)
        pressure_momentum += random.uniform(-0.02, 0.02)
        pressure_momentum = np.clip(pressure_momentum, -0.1, 0.1)
        
        # Random position jitter (more pronounced)
        jitter_x = random.uniform(-2.0, 2.0)
        jitter_y = random.uniform(-2.5, 2.5)

        # Measure glyph bbox using the Draw object
        try:
            x0, y0, x1, y1 = draw.textbbox((0, 0), ch, font=font)
            w = max(1, x1 - x0)
            h = max(1, y1 - y0)
        except Exception:
            w, h, x0, y0 = 20, 30, 0, 0

        # Add padding for effects
        pad = 8
        ch_img = Image.new("RGBA", (w + pad*2, h + pad*2), (255, 255, 255, 0))
        ch_draw = ImageDraw.Draw(ch_img)
        
        # Vary ink color per character only slightly (very small jitter)
        color_variation = tuple(max(0, min(255, c + random.randint(-6, 6))) for c in ink_color)
        
        ch_draw.text((pad - x0, pad - y0), ch, fill=color_variation, font=font)

        # Apply pressure variation (affects thickness and darkness)
        ch_img = apply_pressure_variation(ch_img, pressure)

        # Random rotation with slight scale variation
        angle = random.uniform(-3, 3)
        scale = random.uniform(0.97, 1.03)
        new_size = (int(ch_img.width * scale), int(ch_img.height * scale))
        ch_img = ch_img.resize(new_size, Image.LANCZOS)
        ch_img = ch_img.rotate(angle, resample=Image.BICUBIC, expand=1)

        # Paste using alpha channel as mask
        paste_x = int(cursor_x + jitter_x - pad)
        paste_y = int(y + baseline_shift + jitter_y - pad)
        img.paste(ch_img, (paste_x, paste_y), ch_img)

        # Variable kerning (spacing between letters)
        kerning_var = random.uniform(-2.5, 2.5)
        cursor_x += w + kerning_var


def render_text_block(block, font_path, page_size=(1600, 2000)):
    img = Image.new("RGB", page_size, (255, 255, 255))
    draw = ImageDraw.Draw(img)

    try:
        font_path_expanded = os.path.expanduser(font_path)
        if not os.path.isfile(font_path_expanded):
            raise OSError(f"Font file not found: {font_path_expanded}")
        font = ImageFont.truetype(font_path_expanded, 38)
    except Exception as e:
        print(f"Warning: could not load font '{font_path}'. Falling back to default font. ({e})")
        font = ImageFont.load_default()

    x, y = 100, 100

    # Compute line height robustly
    try:
        ascent, descent = font.getmetrics()
        line_height = ascent + descent + 10
    except Exception:
        try:
            x0, y0, x1, y1 = draw.textbbox((0, 0), "Hg", font=font)
            line_height = (y1 - y0) + 10
        except Exception:
            line_height = 30

    cumulative_drift = 0  # gradual vertical drift across lines

    for line in block["text"].splitlines():
        if not line.strip():
            y += line_height
            continue

        # --- new realism parameters ---
        left_margin_drift = random.uniform(-15, 15)      # line start offset
        cumulative_drift += random.uniform(-2, 2)        # slow vertical drift
        spacing_drift = random.randint(-5, 10)           # uneven line spacing
        baseline_drift = random.uniform(-4, 4)           # slight baseline shake
        
        # Choose a single base ink color for this line (keeps a single pen feel)
        base_blue = random.randint(100, 115)
        base_shift = random.randint(-2, 2)
        ink_variation = (10 + base_shift, 10 + base_shift, base_blue)

        draw_handwritten_line(
            img,
            draw,
            line,
            x + left_margin_drift,
            y + cumulative_drift + baseline_drift,
            font,
            ink_variation
        )
        y += line_height + spacing_drift

    return img


def apply_page_distortion(pil_img):
    """Apply realistic paper warp and subtle rotation."""
    img = np.array(pil_img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    rows, cols, _ = img.shape

    # Slight sinusoidal warp in y-axis (paper not perfectly flat)
    for i in range(rows):
        shift = int(3.0 * math.sin(2 * math.pi * i / 180))
        img[i] = np.roll(img[i], shift, axis=0)

    # Random rotation ±1.5 degrees
    angle = random.uniform(-1.5, 1.5)
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    img = cv2.warpAffine(img, M, (cols, rows), borderValue=(255, 255, 255))

    # Very subtle Gaussian blur for natural ink spread
    img = cv2.GaussianBlur(img, (3, 3), 0)
    
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def add_ink_noise(pil_img):
    """Enhanced ink texture with local variations."""
    img = np.array(pil_img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = gray < 200  # text region mask
    
    # Multi-scale noise for realistic ink texture
    noise = np.random.normal(0, 8, img.shape).astype(np.int16)
    fine_noise = np.random.normal(0, 3, img.shape).astype(np.int16)
    
    combined_noise = noise + fine_noise
    img = np.clip(img + combined_noise * mask[:, :, None], 0, 255).astype(np.uint8)
    
    return Image.fromarray(img)


def add_paper_texture(pil_img, intensity=0.1):
    """Adds realistic paper grain with fiber-like structure."""
    img = np.array(pil_img).astype(np.float32)
    
    # Multi-frequency noise for paper texture
    coarse_noise = np.random.normal(0, 255 * intensity * 1.5, img.shape)
    fine_noise = np.random.normal(0, 255 * intensity * 0.5, img.shape)
    
    # Add subtle paper fibers (directional blur)
    fiber_noise = cv2.GaussianBlur(coarse_noise, (3, 1), 0)
    
    combined_texture = coarse_noise + fine_noise + fiber_noise * 0.3
    img = np.clip(img + combined_texture, 0, 255).astype(np.uint8)
    
    return Image.fromarray(img)


def process_block_to_image(block, font_path):
    """Apply full pipeline with enhanced realism."""
    base_img = render_text_block(block, font_path)
    img = apply_page_distortion(base_img)
    img = add_ink_noise(img)
    img = add_paper_texture(img, 0.08)
    
    return img


def pdf_to_handwritten(pdf_path, output_dir, font_path):
    os.makedirs(output_dir, exist_ok=True)
    blocks = extract_text_blocks(pdf_path)

    for i, block in enumerate(blocks):
        handwritten_img = process_block_to_image(block, font_path)
        out_path = f"{output_dir}/page_{block['page']}_{i}.png"
        handwritten_img.save(out_path)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    pdf_path = "/Users/manikmanavenddram/Text-Ink/pdf/test.pdf"
    output_dir = "output_images"
    font_path = "/Users/manikmanavenddram/Text-Ink/TTF/PatrickHand-Regular.ttf"
    pdf_to_handwritten(pdf_path, output_dir, font_path)