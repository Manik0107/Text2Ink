# .ink

**Intelligent Handwritten Note Generation**

.ink is an advanced document processing engine that transforms standard PDFs and YouTube transcripts into structured, professional handwritten notes. Beyond simple font conversion, it utilizes a sophisticated AI pipeline to analyze content, synthesize logical structures, and generate systemic diagrams that are seamlessly integrated into a realistic handwritten layout.

## Core Capabilities

.ink supports two input sources: **PDF documents** and **YouTube videos** (via transcript extraction). Both are processed through the same intelligent pipeline.

### 1. Intelligent Content Analysis (Agno Agents)
.ink employs a multi-agent AI system:
*   **Notes Agent**: Analyzes raw PDF text to extract key concepts, structuring them into a clean, hierarchical format with substantial headers and bullet points.
*   **Mermaid Agent**: Identifies complex relationships and processes within the text to generate valid Mermaid.js diagram definitions. It follows a strict "Systemic Logic" manifesto, ensuring diagrams represent causality, feedback loops, and inputs/outputs rather than simple associations.

### 2. Systemic Diagram Generation
*   **Vertical Layout Optimization**: The engine prioritizes Top-Down (`graph TD`) layouts to maximize diagram size and readability on portrait pages.
*   **Gigapixel Scaling**: Diagrams are treated as first-class citizens. The engine forces every diagram to upscale to the full width of the page (1400px), ensuring maximum visibility.
*   **Multi-Page Slicing**: If a diagram is too large to fit on a single page after upscaling, the engine automatically slices it into page-sized segments, spanning it across multiple consecutive pages without loss of detail.
*   **Robust Rendering**: Diagram requests use JSON-state encoding to handle special characters and complex syntax reliably.

### 3. Floating Layout Engine
The rendering system treats content as a fluid stream of "Blocks" (Text and Diagrams):
*   **Gap Filling**: If a large diagram must be moved to a new page to fit, the layout engine automatically fills the remaining space on the previous page with subsequent text, eliminating unprofessional vertical gaps.
*   **Strict Pagination**: Content flows continuously across pages, creating a cohesive notebook feel similar to a human-written journal.

### 4. Realistic Handwriting Synthesis
*   **Text Formatting**: Automatically parses Markdown syntax, converting headers to uppercase bold text and standardizing lists with bullet points.
*   **Visual Fidelity**: Incorporates realistic paper textures, ruled lines, and ink variability (though pressure is normalized for legibility) to create high-quality output images.

## Installation

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/Manik0107/.ink.git
    cd .ink
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure Environment**:
    Create a `.env` file and add your API keys (required for the AI Agents):
    ```bash
    OPENROUTER_API_KEY=your_key_here
    ```

## Usage

### Processing PDFs

1.  **Prepare Input**: Place your PDF document in the specified input path (e.g., `pdf/test.pdf`).
2.  **Run the Engine**:
    ```bash
    uv run main.py pdf/your_document.pdf
    ```
3.  **View Output**: The generated handwritten pages and compiled PDF will be saved in the `output_images` directory.

### Processing YouTube Videos

1.  **Get YouTube URL**: Copy the URL of any YouTube video with available transcripts.
2.  **Run the Engine**:
    ```bash
    uv run main.py "https://www.youtube.com/watch?v=VIDEO_ID"
    ```
3.  **View Output**: The transcript will be converted to handwritten notes and saved in the `output_images` directory.

### Command-Line Options

```bash
# Specify custom output directory
uv run main.py input.pdf -o custom_output/

# Use a different handwriting font
uv run main.py "https://youtu.be/VIDEO_ID" -f fonts/custom_font.ttf

# Combine options
uv run main.py input.pdf -o notes/ -f fonts/script.ttf
```

**Note**: YouTube videos must have transcripts enabled (either auto-generated or manual). The tool automatically detects whether the input is a PDF or YouTube URL.

## Technical Architecture

The system operates as a linear pipeline:
1.  **Extraction**: `fitz` (PyMuPDF) extracts raw text from the source PDF.
2.  **Orchestration**: `agent.py` invokes the AI Agents to transform raw text into a Structured Note Object containing text blocks and Mermaid diagram codes.
3.  **Rendering**: `main.py` processes the Note Object:
    *   **Text Rendering**: Uses Pillow (PIL) to draw text with randomized handwriting fonts on a ruled background.
    *   **Diagram Rendering**: Fetches rendered diagrams from `mermaid.ink`, applies upscale/slice logic, and composites them onto the page.
4.  **Assembly**: Individual page images are compiled into a final `handwritten_notes.pdf`.

## License

This project is licensed under the Apache License 2.0.
