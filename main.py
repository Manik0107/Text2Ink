import fitz
from PIL import Image, ImageDraw, ImageFont
import random, math, os
import cv2
import numpy as np
from dotenv import load_dotenv
from agent import generate_simple_notes, generate_mermaid_code

load_dotenv()


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
    
    # Balanced pressure:
    # We avoid erosion because it breaks thin fonts.
    # We only use subtle lightening for low pressure.
    
    if pressure < 0.9:
        # Just lighten slightly, don't erode
        factor = 0.5 * (1 - pressure) # mild lightening
        img_array[:, :, :3] = np.clip(img_array[:, :, :3] + (255 - img_array[:, :, :3]) * factor, 0, 255).astype(np.uint8)
        
    elif pressure > 1.1:
        # Thicken slightly for heavy pressure
        kernel_size = 2
        # Use a milder dilation structure (cross instead of block)
        kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
        alpha = cv2.dilate(alpha, kernel, iterations=1)
        
        # Darken slightly
        dark_factor = 0.90 # only 10% darker max
        img_array[:, :, :3] = (img_array[:, :, :3] * dark_factor).astype(np.uint8)
    
    img_array[:, :, 3] = alpha
    return Image.fromarray(img_array)





def draw_handwritten_line(img, draw, text, x, y, font, ink_color, slant_angle=0, line_slope=0):
    """
    Draw text with realistic human variations.
    x, y: coordinates of the starting BASELINE.
    """
    cursor_x = x
    
    # Get font metrics for vertical alignment
    try:
        ascent, descent = font.getmetrics()
    except Exception:
        ascent, descent = 30, 10 # Fallback
    
    font_height = ascent + descent

    # Organic baseline fluctuation (sine wave + noise)
    baseline_freq = random.uniform(0.005, 0.012)
    baseline_amp = random.uniform(1.5, 3.0) # Reduced slightly for better ruled line adhesion
    baseline_phase = random.uniform(0, math.pi * 2)

    word_gap_base = 14        
    word_gap_jitter = 3
    
    # Constant pen pressure
    pressure = 1.0

    for i, ch in enumerate(text):
        # Calculate local y-offset based on line slope and organic baseline
        dist_from_start = cursor_x - x
        slope_offset = dist_from_start * line_slope
        
        # Sinusoidal baseline drift
        sine_offset = math.sin(cursor_x * baseline_freq + baseline_phase) * baseline_amp
        
        if ch == " ":
            cursor_x += word_gap_base + random.uniform(-word_gap_jitter, word_gap_jitter)
            continue
            
        # No pressure evolution
        pressure = 1.0
        
        # Reduced random positional jitter
        jitter_x = random.uniform(-0.5, 0.5)
        jitter_y = random.uniform(-1.0, 1.0) # Smaller vertical jitter

        # Get character width
        try:
            x0, y0, x1, y1 = draw.textbbox((0, 0), ch, font=font)
            w = max(1, x1 - x0)
        except Exception:
            w, x0 = 20, 0

        pad = 20 # Larger padding to accommodate rotation without clipping
        
        # Create character image large enough to hold the full font height + padding
        # We draw relative to a fixed baseline position to preserve alignment.
        # If we draw at y=pad, standard PIL draws top-left at pad. Baseline is at pad + ascent.
        
        ch_img_w = w + pad * 2
        ch_img_h = font_height + pad * 2
        
        ch_img = Image.new("RGBA", (ch_img_w, ch_img_h), (255, 255, 255, 0))
        ch_draw = ImageDraw.Draw(ch_img)
        
        # Vary ink color slightly
        color_variation = tuple(max(0, min(255, c + random.randint(-8, 8))) for c in ink_color)
        
        # Draw the character. 
        # We shift X by -x0 to tightly crop/fit the left side (standard spacing).
        # We DO NOT shift Y by -y0. We draw at a fixed Y=pad.
        # This puts the top-left of the glyph's layout box at (pad-x0, pad).
        # The baseline will be at y = pad + ascent.
        ch_draw.text((pad - x0, pad), ch, fill=color_variation, font=font)

        ch_img = apply_pressure_variation(ch_img, pressure)

        # Consistent slant
        char_angle = slant_angle + random.uniform(-2, 2)
        
        scale_x = random.uniform(0.95, 1.05)
        scale_y = random.uniform(0.95, 1.05)
        new_size = (int(ch_img_w * scale_x), int(ch_img_h * scale_y))
        
        ch_img = ch_img.resize(new_size, Image.LANCZOS)
        ch_img = ch_img.rotate(char_angle, resample=Image.BICUBIC, expand=True)

        # Paste position:
        # We want the baseline of the character to align with:
        # y (base line) + offsets.
        # Inside ch_img, baseline is at (roughly) pad + ascent (scaled).
        # To align baselines:
        # paste_y + (pad + ascent)*scale_y = target_baseline
        # paste_y = target_baseline - (pad + ascent)*scale_y
        
        # Note: rotation changes the center, but for small angles we can approximate/ignore or center-pivot.
        # rotate(expand=True) changes the image size.
        # It pivots around the center.
        # original center: (w_old/2, h_old/2). baseline dist from center: (pad+ascent) - h_old/2.
        # rotated center is at (w_new/2, h_new/2).
        # new baseline is roughly center + (baseline_dist_from_center).
        
        # For simplicity with small angles, we can assume the vertical offset is roughly preserved relative to top
        # or just use the center alignment.
        
        # Let's trust the "pad + ascent" as the primary anchor logic and adjust for scale.
        
        anchor_y_in_ch_img = (pad + ascent) * scale_y
        
        paste_x = int(cursor_x + jitter_x - pad)
        target_baseline = y + slope_offset + sine_offset + jitter_y
        paste_y = int(target_baseline - anchor_y_in_ch_img)
        
        img.paste(ch_img, (paste_x, paste_y), ch_img)

        # Variable kerning
        kerning_var = random.uniform(-1.0, 1.5)
        cursor_x += w + kerning_var


def draw_ruled_lines(draw, width, height, spacing=60, top_margin=100):
    """Draws ruled lines on the page background."""
    # Light blue/grey color for lines
    line_color = (180, 180, 200)
    
    # Draw horizontal lines
    y = top_margin
    while y < height - 50:
        # Slight line imperfections
        line_width = random.randint(width - 20, width)
        drift_y = random.uniform(-0.5, 0.5)
        
        draw.line([(0, y + drift_y), (line_width, y + drift_y)], fill=line_color, width=2)
        y += spacing
        
    margin_x = 100
    draw.line([(margin_x, 0), (margin_x, height)], fill=(200, 100, 100), width=2)


def wrap_text(text, font, max_width):
    """Wraps text to fit within max_width."""
    lines = []
    # If text has newlines, handle them first
    paragraphs = text.splitlines()
    
    for paragraph in paragraphs:
        if not paragraph:
            lines.append("")
            continue
            
        words = paragraph.split(' ')
        current_line = []
        
        for word in words:
            # Check width of current line + word
            test_line = ' '.join(current_line + [word])
            w = font.getlength(test_line)
            
            if w <= max_width:
                current_line.append(word)
            else:
                # If the line is not empty, push it and start a new one
                if current_line:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                else:
                    # If the word itself is too long, just put it on the line (rare case)
                    lines.append(word)
                    current_line = []
        
        if current_line:
            lines.append(' '.join(current_line))

    return lines



def create_ruled_page(page_size=(1600, 2000)):
    """Creates a blank ruled page."""
    img = Image.new("RGB", page_size, (255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    TOP_MARGIN = 150
    RULE_SPACING = 70
    
    draw_ruled_lines(draw, img.width, img.height, spacing=RULE_SPACING, top_margin=TOP_MARGIN)
    return img, draw, TOP_MARGIN, RULE_SPACING


import httpx
import base64

def get_mermaid_image(mermaid_code):
    """Fetches a PNG of the mermaid diagram from mermaid.ink."""
    graphbytes = mermaid_code.encode("utf8")
    base64_bytes = base64.urlsafe_b64encode(graphbytes)
    base64_string = base64_bytes.decode("ascii")
    url = "https://mermaid.ink/img/" + base64_string
    
    try:
        response = httpx.get(url, timeout=10.0)
        response.raise_for_status()
        img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    except Exception as e:
        print(f"Error fetching mermaid diagram: {e}")
        return None


def render_full_text(full_text, font_path, page_size=(1600, 2000)):
    """Renders text across multiple pages with formatting and diagrams."""
    
    # 1. Load Font
    try:
        font_path_expanded = os.path.expanduser(font_path)
        if not os.path.isfile(font_path_expanded):
            raise OSError(f"Font file not found: {font_path_expanded}")
        font_size = int(70 * 0.75)  
        font = ImageFont.truetype(font_path_expanded, font_size)
    except Exception as e:
        print(f"Warning: could not load font '{font_path}'. Falling back to default font. ({e})")
        font = ImageFont.load_default()

    pages = []
    
    # Start first page
    current_img, current_draw, TOP_MARGIN, RULE_SPACING = create_ruled_page(page_size)
    current_baseline_y = TOP_MARGIN + RULE_SPACING
    
    # 2. Process Content Blocks (Text vs. Mermaid)
    # Split by mermaid blocks
    parts = full_text.split("```mermaid")
    
    for i, part in enumerate(parts):
        if i > 0: # This part starts with mermaid code
            if "```" in part:
                mermaid_code, text_content = part.split("```", 1)
                
                # Render and paste mermaid diagram
                mermaid_img = get_mermaid_image(mermaid_code.strip())
                if mermaid_img:
                    # SMART SCALING LOGIC
                    # 1. Start with upscale for quality (2.0x for high visibility)
                    scale_factor = 2.0
                    new_w = int(mermaid_img.width * scale_factor)
                    new_h = int(mermaid_img.height * scale_factor)
                    mermaid_img = mermaid_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                    
                    # 2. Check Constraints
                    max_w = page_size[0] - 200 # 100 margin each side
                    max_h = page_size[1] - 300 # Top/Bottom margins
                    
                    # 3. Scale Width if needed
                    if mermaid_img.width > max_w:
                        ratio = max_w / mermaid_img.width
                        new_w = max_w
                        new_h = int(mermaid_img.height * ratio)
                        mermaid_img = mermaid_img.resize((int(new_w), int(new_h)), Image.Resampling.LANCZOS)
                        
                    # 4. Check Height / Pagination
                    # If it fits on current page?
                    if current_baseline_y + mermaid_img.height > current_img.height - 100:
                        # Move to new page
                         pages.append(current_img)
                         current_img, current_draw, TOP_MARGIN, RULE_SPACING = create_ruled_page(page_size)
                         current_baseline_y = TOP_MARGIN + RULE_SPACING
                    
                    # 5. Check if it fits FULL page (rare but possible for mindmaps)
                    if mermaid_img.height > max_h:
                         ratio = max_h / mermaid_img.height
                         new_h = max_h
                         new_w = int(mermaid_img.width * ratio)
                         mermaid_img = mermaid_img.resize((int(new_w), int(new_h)), Image.Resampling.LANCZOS)
                    
                    # Center image
                    offset_x = (page_size[0] - mermaid_img.width) // 2
                    current_img.paste(mermaid_img, (offset_x, int(current_baseline_y)), mask=mermaid_img.split()[3] if 'A' in mermaid_img.getbands() else None)
                    current_baseline_y += mermaid_img.height + RULE_SPACING
            else:
                # Malformed block, treat as text
                text_content = part
        else:
            text_content = part

        if not text_content.strip():
            continue

        # 3. Process Text Lines (Markdown Parsing)
        # We split by newlines to respect paragraphs and lists
        raw_lines = text_content.split('\n')
        
        for raw_line in raw_lines:
            stripped_line = raw_line.strip()
            
            # Paragraph spacing check
            if not stripped_line:
                current_baseline_y += RULE_SPACING // 2 # Half line for para break
                continue
                
            # Indentation/List detection
            indent_level = 0
            prefix_width = 0
            clean_line = stripped_line
            
            if raw_line.startswith('- ') or raw_line.startswith('* '):
                indent_level = 1
                prefix_width = 40
                clean_line = raw_line[2:]
            elif raw_line.startswith('  - ') or raw_line.startswith('  * '):
                indent_level = 2
                prefix_width = 80
                clean_line = raw_line[4:]
            elif raw_line.startswith('#'): # Headings
                 clean_line = raw_line.lstrip('#').strip()
            
            # Wrap based on effective width
            x_start = 120 + prefix_width
            max_width = page_size[0] - x_start - 100
            
            wrapped_lines = wrap_text(clean_line, font, max_width)

            for sub_line in wrapped_lines:
                # Check page space
                if current_baseline_y > current_img.height - 100:
                    pages.append(current_img)
                    current_img, current_draw, TOP_MARGIN, RULE_SPACING = create_ruled_page(page_size)
                    current_baseline_y = TOP_MARGIN + RULE_SPACING
                
                # Draw Bullet point only on first subline
                if sub_line == wrapped_lines[0] and indent_level > 0:
                     bullet_x = 120 + (indent_level * 30) - 15
                     bullet_y = current_baseline_y - 10
                     current_draw.ellipse([bullet_x-3, bullet_y-3, bullet_x+3, bullet_y+3], fill="black")

                # --- Realism parameters ---
                left_margin_drift = random.uniform(-2, 5) 
                baseline_drift = random.uniform(-1, 1)           
                line_float = random.uniform(-8, -2) 
                
                base_blue = random.randint(100, 115)
                base_shift = random.randint(-2, 2)
                ink_variation = (10 + base_shift, 10 + base_shift, base_blue)

                line_slope = random.uniform(-0.01, 0.01) 
                global_slant = random.uniform(-12, 2)
                
                draw_handwritten_line(
                    current_img,
                    current_draw,
                    sub_line,
                    x_start + left_margin_drift,
                    current_baseline_y + line_float + baseline_drift,
                    font,
                    ink_variation,
                    slant_angle=global_slant,
                    line_slope=line_slope
                )
                current_baseline_y += RULE_SPACING
    
    # Append the last page
    pages.append(current_img)
    return pages


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


def process_image_post_effects(base_img):
    """Apply textures to the generated base image."""
    img = apply_page_distortion(base_img)
    # img = add_ink_noise(img)       # Disabled to remove grains
    # img = add_paper_texture(img, 0.08) # Disabled to remove grains
    return img


def pdf_to_handwritten(pdf_path, output_dir, font_path):
    os.makedirs(output_dir, exist_ok=True)
    blocks = extract_text_blocks(pdf_path)
    
    # Aggregate all text
    full_text = ""
    for b in blocks:
        full_text += b["text"] + "\n\n" # Add some internal spacing between blocks

    print(f"Total Text Length (Original): {len(full_text)} characters")

    # Generate Structured Notes (with Inline Diagrams found by the Agent)
    print("Generating structured notes with inline diagrams...")
    # This function now orchestrates both the notes and the mermaid diagrams
    final_content = generate_simple_notes(full_text) 
    print(f"Final Content Length: {len(final_content)} characters")
    
    # Render pages using the notes
    raw_pages = render_full_text(final_content, font_path)
    
    final_images = []
    for i, page_img in enumerate(raw_pages):
        final_img = process_image_post_effects(page_img)
        out_path = f"{output_dir}/page_{i}.png"
        final_img.save(out_path)
        final_images.append(final_img)
        print(f"Saved {out_path}")
        
    # Save as PDF
    if final_images:
        pdf_path = f"{output_dir}/handwritten_notes.pdf"
        final_images[0].save(
            pdf_path, "PDF", resolution=100.0, save_all=True, append_images=final_images[1:]
        )
        print(f"Saved PDF to {pdf_path}")


if __name__ == "__main__":
    pdf_path = "pdf/test.pdf"
    output_dir = "output_images"
    font_path = "TTF/PatrickHand-Regular.ttf"
    pdf_to_handwritten(pdf_path, output_dir, font_path)