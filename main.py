import fitz
from PIL import Image, ImageDraw, ImageFont
from collections import deque
import random, math, os
import re
import cv2
import numpy as np
import httpx
import base64
import json
import argparse
from dotenv import load_dotenv
from agent import generate_simple_notes, generate_mermaid_code
from youtube_extractor import get_youtube_text

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
    """Fetches a PNG of the mermaid diagram from mermaid.ink using JSON encoding."""
    state = {
        "code": mermaid_code,
        "mermaid": {
            "theme": "neutral",
            "securityLevel": "loose"
        }
    }
    json_str = json.dumps(state)
    state_bytes = json_str.encode('utf-8')
    base64_bytes = base64.urlsafe_b64encode(state_bytes)
    base64_string = base64_bytes.decode('ascii')
    
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


def place_diagram_on_page(mermaid_img, img, y, page_size, spacing):
    """
    Helper to place a pre-rendered mermaid image.
    STRICTLY enforces that the image fits within the page bottom margin.
    Resizes the image if necessary.
    """
    bottom_limit = page_size[1] - 100
    available_h = bottom_limit - y
    
    # 1. Strict Fit Check
    if mermaid_img.height > available_h:
        # Resize to fit
        if available_h < 100: # Safety for nearly zero space (shouldn't happen with MIN checks but safety)
             return y # Cannot place
             
        ratio = available_h / mermaid_img.height
        new_w = int(mermaid_img.width * ratio)
        new_h = int(available_h)
        mermaid_img = mermaid_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # 2. Centering & Paste
    offset_x = (page_size[0] - mermaid_img.width) // 2
    img.paste(mermaid_img, (offset_x, int(y)), mask=mermaid_img.split()[3] if 'A' in mermaid_img.getbands() else None)
    
    return y + mermaid_img.height + spacing


def render_full_text(full_text, font_path):
    """
    Renders text and diagrams with a 'Floating Figure' layout engine.
    If a diagram doesn't fit, it moves to the next page, and the current page
    is filled with the subsequent text (if available) to avoid vertical gaps.
    """
    print("Starting Floating Layout Render...")
    
    # 1. Parse content into a queue of items
    items = []
    # Split by mermaid blocks
    parts = re.split(r"(```mermaid.*?```)", full_text, flags=re.DOTALL)
    for part in parts:
        if part.strip().startswith("```mermaid"):
            code = part.replace("```mermaid", "").replace("```", "").strip()
            if code:
                 items.append({"type": "mermaid", "content": code})
        elif part.strip():
             items.append({"type": "text", "content": part.strip()})
             
    queue = deque(items)
    
    # 2. Setup Page
    page_size = (1600, 2000)
    pages = []
    current_img, current_draw, TOP_MARGIN, RULE_SPACING = create_ruled_page(page_size)
    current_baseline_y = TOP_MARGIN + RULE_SPACING
    # Font setup
    font_path_expanded = os.path.expanduser(font_path)
    if not os.path.isfile(font_path_expanded):
         print(f"Warning: Font not found, using default.")
         font = ImageFont.load_default()
         font_height = 40
    else:
         font_size = int(RULE_SPACING * 0.75)
         font = ImageFont.truetype(font_path_expanded, font_size)
         ascent, descent = font.getmetrics()
         font_height = ascent + descent

    pending_diagram = None # To hold a diagram that needs to move to next page
    
    while queue or pending_diagram:
        
        # A. Check if we need a new page immediately (e.g., extremely full)
        if current_baseline_y > page_size[1] - 100:
             pages.append(current_img)
             current_img, current_draw, TOP_MARGIN, RULE_SPACING = create_ruled_page(page_size)
             current_baseline_y = TOP_MARGIN + RULE_SPACING
             
             # If we had a pending diagram, place it now at TOP of new page
             if pending_diagram:
                 current_baseline_y = place_diagram_on_page(pending_diagram, current_img, current_baseline_y, page_size, RULE_SPACING)
                 pending_diagram = None
                 continue 

        # B. If we have a pending diagram but are still on the OLD page (gap filling mode)
        #    We try to squeeze text in. But if queue is empty or next item is diagram, force new page.
        if pending_diagram:
             if not queue or queue[0]["type"] == "mermaid":
                 # Cannot fill gap. Force new page.
                 pages.append(current_img)
                 current_img, current_draw, TOP_MARGIN, RULE_SPACING = create_ruled_page(page_size)
                 current_baseline_y = TOP_MARGIN + RULE_SPACING
                 current_baseline_y = place_diagram_on_page(pending_diagram, current_img, current_baseline_y, page_size, RULE_SPACING)
                 pending_diagram = None
                 continue
                 
             # Else: Next item is TEXT. Fill the gap!
             item = queue.popleft()
             # Calculate remaining lines
             remaining_height = (page_size[1] - 100) - current_baseline_y
             lines_capacity = int(remaining_height / RULE_SPACING)
             
             if lines_capacity <= 0: 
                 # Cannot fill any more lines. Force new page for diagram.
                 queue.appendleft(item) # Put back
                 
                 pages.append(current_img)
                 current_img, current_draw, TOP_MARGIN, RULE_SPACING = create_ruled_page(page_size)
                 current_baseline_y = TOP_MARGIN + RULE_SPACING
                 current_baseline_y = place_diagram_on_page(pending_diagram, current_img, current_baseline_y, page_size, RULE_SPACING)
                 pending_diagram = None
                 continue 
                 
             # Draw only what fits
             remaining_text, new_y = draw_text_chunk(current_img, current_draw, item["content"], current_baseline_y, font, RULE_SPACING, page_size, max_lines=lines_capacity)
             current_baseline_y = new_y
             
             if remaining_text:
                 # We filled the page, but text remains. 
                 # Put remainder back at HEAD of queue
                 queue.appendleft({"type": "text", "content": remaining_text})
                 # Loop will hit A/B logic, force new page for diagram
             
             continue # Loop again to trigger new page logic

        # C. Normal Processing (No pending diagram)
        item = queue.popleft()
        
        if item["type"] == "mermaid" or item["type"] == "mermaid_image":
             # 1. Get Image
             if item["type"] == "mermaid":
                 mermaid_img = get_mermaid_image(item["content"])
                 if not mermaid_img: continue
             else:
                 mermaid_img = item["content"] # Pre-rendered slice

             # 2. Force Upscale to Full Page Width (if not already sliced check)
             # If it's a raw mermaid code, we ALWAYS upscale. 
             # If it's a slice (mermaid_image), we assume it's already width-adjusted, but double check?
             # Let's upscale everything to ensure consistency.
             target_w = page_size[0] - 200 # Margins
             
             if mermaid_img.width != target_w:
                 ratio = target_w / mermaid_img.width
                 new_h = int(mermaid_img.height * ratio)
                 mermaid_img = mermaid_img.resize((target_w, new_h), Image.Resampling.LANCZOS)
             
             # 3. Check for Massive Height (Multi-Page Slicing)
             MAX_SLICE_HEIGHT = 1600 
             
             if mermaid_img.height > MAX_SLICE_HEIGHT:
                 # SPLIT IT!
                 total_h = mermaid_img.height
                 num_slices = math.ceil(total_h / MAX_SLICE_HEIGHT)
                 print(f"Diagram too tall ({total_h}px). Splitting into {num_slices} pages...")
                 
                 slices = []
                 for i in range(num_slices):
                     top = i * MAX_SLICE_HEIGHT
                     bottom = min((i + 1) * MAX_SLICE_HEIGHT, total_h)
                     # Crop: (left, top, right, bottom)
                     crop_box = (0, top, mermaid_img.width, bottom)
                     slice_img = mermaid_img.crop(crop_box)
                     slices.append(slice_img)
                 
                 # Push slices back to HEAD of queue in reverse order
                 # So slice 0 is processed next, then slice 1...
                 for s in reversed(slices):
                     queue.appendleft({"type": "mermaid_image", "content": s})
                     
                 continue # Loop again to pick up the first slice

             # 4. Standard Placement Logic (for a single slice/image that fits ON A PAGE)
             MIN_VISIBLE_HEIGHT = 1000
             bottom_limit = page_size[1] - 100
             available_h = bottom_limit - current_baseline_y
             
             if available_h < MIN_VISIBLE_HEIGHT:
                 # Too small for this diagram. Defer to next page.
                 pending_diagram = mermaid_img 
             else:
                 # Check if it fits the *current* gap
                 if mermaid_img.height > available_h:
                      # If it doesn't fit mostly, defer it (unless it's smaller than min visible?)
                      # Since we sliced it to MAX_SLICE_HEIGHT, it might fit on a NEW page, but not here.
                      pending_diagram = mermaid_img
                 else:
                      # Fits here nicely
                      current_baseline_y = place_diagram_on_page(mermaid_img, current_img, current_baseline_y, page_size, RULE_SPACING)
                 
        else: # Text
             remaining_text, new_y = draw_text_chunk(current_img, current_draw, item["content"], current_baseline_y, font, RULE_SPACING, page_size)
             current_baseline_y = new_y
             if remaining_text:
                 # Didn't fit. Put remainder back. Logic A will handle new page.
                 queue.appendleft({"type": "text", "content": remaining_text})

    pages.append(current_img)
    return pages


def draw_text_chunk(img, draw, text, start_y, font, rule_spacing, page_size, max_lines=None):
    """
    Draws text until space runs out or text ends.
    Returns (remaining_text, new_y).
    """
    # 1. Pre-process: Handle literal escapes and split lines
    text = text.replace('\\n', '\n')
    
    # 2. Parse Markdown Structure
    processed_lines = []
    
    # Split by actual newlines to respect structure
    raw_lines = text.split('\n')
    
    for line in raw_lines:
        line = line.strip()
        if not line:
            # Empty line = Paragraph break (simulate with empty string)
            processed_lines.append("") 
            continue
            
        if line.startswith('#'):
            # Heading -> Uppercase
            clean = line.lstrip('#').strip().upper()
            processed_lines.append(clean)
            processed_lines.append("") # Add space after heading
        elif line.startswith('- ') or line.startswith('* ') or line.startswith('\\- '):
            # List -> Bullet (Handle \- artifact)
            clean = "• " + line.replace('\\-', '').lstrip('-').lstrip('*').strip()
            processed_lines.append(clean)
        else:
            # Normal text
            processed_lines.append(line)
            
    # 3. Wrapping & Drawing Loop
    drawn_count = 0
    y = start_y
    bottom_limit = page_size[1] - 100
    
    # We must iterate through processed lines, wrap each one, and draw
    # If we run out of space, we must reconstruct the remaining text correctly.
    
    for i, p_line in enumerate(processed_lines):
        if not p_line:
             # Just a spacing line (paragraph break)
             if max_lines and drawn_count >= max_lines:
                 # Reconstruct remaining
                 remaining = "\n".join(processed_lines[i:])
                 return remaining, y
             
             if y > bottom_limit:
                 remaining = "\n".join(processed_lines[i:])
                 return remaining, y
                 
             y += rule_spacing // 2 # Half space for para break
             continue
             
        # Wrap this content line
        # Check if it's a bullet to indent
        indent = 40 if p_line.startswith("• ") else 0
        wrapped = wrap_text(p_line, font, page_size[0] - 300 - indent)
        
        for w_line in wrapped:
            if max_lines and drawn_count >= max_lines:
                 # Current p_line is partially drawn? No, simple logic: return current p_line start
                 # But we might have drawn half of wrapped? 
                 # Complex. Let's simplify: If ANY part of p_line fails, return p_line + rest
                 remaining = "\n".join(processed_lines[i:])
                 return remaining, y
            
            if y > bottom_limit:
                 remaining = "\n".join(processed_lines[i:])
                 return remaining, y
            
            draw_x = 150 + indent
            draw_handwritten_line(img, draw, w_line, draw_x, y, font, (20, 20, 100))
            y += rule_spacing
            drawn_count += 1
            
    return None, y


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


def text_to_handwritten(text, output_dir, font_path):
    """Core function to convert any text to handwritten notes."""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Total Text Length (Original): {len(text)} characters")

    # Generate Structured Notes (with Inline Diagrams found by the Agent)
    print("Generating structured notes with inline diagrams...")
    final_content = generate_simple_notes(text) 
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
        pdf_output_path = f"{output_dir}/handwritten_notes.pdf"
        final_images[0].save(
            pdf_output_path, "PDF", resolution=100.0, save_all=True, append_images=final_images[1:]
        )
        print(f"Saved PDF to {pdf_output_path}")


def pdf_to_handwritten(pdf_path, output_dir, font_path):
    """Convert PDF to handwritten notes."""
    print(f"Processing PDF: {pdf_path}")
    blocks = extract_text_blocks(pdf_path)
    
    # Aggregate all text
    full_text = ""
    for b in blocks:
        full_text += b["text"] + "\n\n"
    
    text_to_handwritten(full_text, output_dir, font_path)


def youtube_to_handwritten(youtube_url, output_dir, font_path):
    """Convert YouTube video transcript to handwritten notes."""
    print(f"Processing YouTube video: {youtube_url}")
    
    try:
        # Extract transcript from YouTube
        transcript_text = get_youtube_text(youtube_url)
        text_to_handwritten(transcript_text, output_dir, font_path)
    except Exception as e:
        print(f"Error processing YouTube video: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Text2Ink - Convert PDFs or YouTube videos to handwritten notes"
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=None,
        help="Input file path (PDF) or YouTube URL"
    )
    
    parser.add_argument(
        "-f", "--font",
        default="TTF/PatrickHand-Regular.ttf",
        help="Path to handwriting font file (default: TTF/PatrickHand-Regular.ttf)"
    )
    
    # Hardcoded output directory
    output_dir = "output"
    
    args = parser.parse_args()
    
    # Auto-detect input type
    input_value = args.input

    # Interactive mode if no argument provided
    if input_value is None:
        print("\nWelcome to .ink - Intelligent Handwritten Notes")
        print("-----------------------------------------------")
        input_value = input("Please enter a YouTube URL or PDF file path: ").strip()
        # Clean up quotes if user pasted them
        input_value = input_value.strip('"').strip("'")
    
    if not input_value:
        print("No input provided.")
        exit(0)
    
    print(f"Output will be saved to: {output_dir}")
    
    if "youtube.com" in input_value or "youtu.be" in input_value:
        # YouTube URL detected
        youtube_to_handwritten(input_value, output_dir, args.font)
    elif os.path.isfile(input_value) and input_value.lower().endswith('.pdf'):
        # PDF file detected
        pdf_to_handwritten(input_value, output_dir, args.font)
    else:
        print(f"Error: Invalid input '{input_value}'.")
        print("Please provide either:")
        print("  - A valid PDF file path")
        print("  - A YouTube URL (youtube.com or youtu.be)")
        exit(1)