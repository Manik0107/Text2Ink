import re
import os
from agno.agent import Agent
from agno.models.openrouter import OpenRouter
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

def get_notes_agent():
    """Configures and returns the Agno agent for plain notes."""
    return Agent(
        model=OpenRouter(id="google/gemini-2.0-flash-001"),
        description="You are an expert note-taker. You convert complex text into simple, easy-to-understand notes.",
        instructions=[
            "Read the provided text carefully.",
            "Summarize the key points into detailed notes.",
            "Structure the notes with clear `# Headings` for every major topic.",
            "Use Markdown formatting:",
            "  - Use bullet points (`- `) for lists.",
            "  - Use double newlines to separate paragraphs.",
            "  - Do NOT use bold or italics as the handwriting renderer ignores them.",
            "**DIAGRAM PLACEMENT**:",
            "If a section explains a complex concept/process that needs visualization, insert a placeholder exactly where the diagram should appear:",
            "  `<<MINDMAP: brief description of specific topic to map>>`",
            "Focus purely on the text content summaries.",
            "Output ONLY the notes with the placeholders."
        ],
        markdown=True, 
    )

def get_mermaid_agent():
    """Configures and returns the Agno agent for Mermaid diagrams."""
    return Agent(
        model=OpenRouter(id="google/gemini-2.0-flash-001"),
        description="You are an expert Diagram Generator. You analyze text and create Mermaid.js flowcharts.",
        instructions=[
            "Analyze the text and SELECT THE BEST DIAGRAM TYPE from the following:",
            "1. **System/Block Diagram** (`graph TD`): Vertical layout for clean structure/hierarchy. (Preferred Default)",
            "2. **Causal Graph** (`graph TD`): Vertical logic chains (Avoid LR to prevent shrinking).",
            "3. **Layered Model** (`subgraph` in `graph TD`): Stacked layers (Data vs Reality).",
            "4. **State Machine** (`stateDiagram-v2`): Vertical state transitions.",
            "5. **Entity Table** (`classDiagram`): Strict rules/components.",
            "6. **Narrative Flow** (`graph TD`): Ordered steps down the page.",
            "**RULES**:",
            "  - **Verticality**: ALWAYS prefer `TD` (Top-Down) over `LR` (Left-Right) to maximize size.",
            "  - **Safety**: Quote labels with special chars: `id[\"Label (Info)\"]`.",
            "  - **Purpose**: Root/Start must be the Goal.",
            "  - **Logic**: Show Inputs -> Process -> Outputs.",
            "  - **Feedback**: Include loops if adaptive.",
            "  - **Minimalism**: Keywords only (1-3 words).",
            "Start with `%%{init: {'theme': 'neutral'} }%%`.",
            "Output ONLY the raw Mermaid code.",
            "If no clear, complex system exists, return 'NO_DIAGRAM'."
        ],
        markdown=False,
    )

def generate_structured_notes(text):
    """Generates notes with inline diagrams using an orchestrator pattern."""
    notes_agent = get_notes_agent()
    mermaid_agent = get_mermaid_agent()
    
    print("Generating comprehensive notes structure...")
    try:
        notes_response = notes_agent.run(f"Create structured notes from:\n\n{text}")
        notes_content = notes_response.content
    except Exception as e:
        print(f"Error generating notes: {e}")
        return text

    # Find all mindmap placeholders
    placeholders = re.findall(r"<<MINDMAP: (.*?)>>", notes_content)
    
    for topic in placeholders:
        print(f"Generating diagram for: {topic}...")
        try:
            # We pass the full text as context, but ask to focus on the specific topic
            diag_response = mermaid_agent.run(
                f"Context: {text[:2000]}...\n\n" # Pass relevant context (truncated if needed)
                f"Task: Generate a mindmap specifically for the topic: '{topic}'."
            )
            mermaid_code = diag_response.content.strip()
            
            # Clean up code
            mermaid_code = mermaid_code.replace("```mermaid", "").replace("```", "").strip()
            
            if "NO_DIAGRAM" not in mermaid_code and len(mermaid_code) > 10:
                replacement = f"\n```mermaid\n{mermaid_code}\n```\n"
            else:
                replacement = "" # Remove placeholder if failed
                
            notes_content = notes_content.replace(f"<<MINDMAP: {topic}>>", replacement)
            
        except Exception as e:
            print(f"Error generating inline diagram: {e}")
            notes_content = notes_content.replace(f"<<MINDMAP: {topic}>>", "")

    return notes_content

def generate_simple_notes(text):
    # Deprecated by generate_structured_notes but kept for compatibility if needed
    return generate_structured_notes(text)

def generate_mermaid_code(text):
     # Deprecated
     pass
