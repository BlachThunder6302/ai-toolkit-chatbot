import os
import docx
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

MEDIA_DIR = "media"

def paragraph_has_page_break(paragraph):
    """Check if paragraph XML contains a page break."""
    if not paragraph._p:
        return False

    for child in paragraph._p.iter():
        if child.tag == qn('w:br') and child.get(qn('w:type')) == 'page':
            return True
    return False


def convert_run_formatting(run):
    """Convert bold/italic text to Markdown equivalents."""
    text = run.text or ""

    if run.bold:
        text = f"**{text}**"
    if run.italic:
        text = f"*{text}*"

    return text


def convert_paragraph_to_md(paragraph):
    """Convert a paragraph to Markdown with formatting."""
    if paragraph.style.name.startswith("Heading"):
        level = paragraph.style.name[-1]  # '1', '2', ...
        return f"{'#' * int(level)} {paragraph.text.strip()}"

    md_text = "".join(convert_run_formatting(run) for run in paragraph.runs)

    # Bullet lists
    if paragraph.style.name in ("List Paragraph", "List Bullet"):
        return f"- {md_text}"

    # Numbered lists
    if "List Number" in paragraph.style.name:
        return f"1. {md_text}"

    return md_text.strip()


def extract_images(doc, doc_slug):
    """Extract images to media folder."""
    os.makedirs(os.path.join(MEDIA_DIR, doc_slug), exist_ok=True)

    img_map = []
    idx = 1

    for shape in doc.inline_shapes:
        if shape._inline.graphic.graphicData.pic:
            img_name = f"img-{idx}.png"
            img_path = os.path.join(MEDIA_DIR, doc_slug, img_name)

            with open(img_path, "wb") as f:
                f.write(shape._inline.graphic.graphicData.pic.blipFill.blip.embed_part.blob)

            img_map.append(img_path)
            idx += 1

    return img_map


def convert_docx_to_markdown(input_path, output_path):
    print(f"\n→ Processing DOCX: {os.path.basename(input_path)}")

    doc = docx.Document(input_path)
    doc_slug = os.path.splitext(os.path.basename(input_path))[0]

    images = extract_images(doc, doc_slug)

    md_lines = []
    page_num = 1
    md_lines.append(f"## Page {page_num}\n")

    img_index = 0

    for para in doc.paragraphs:

        # Page break detection
        if paragraph_has_page_break(para):
            page_num += 1
            md_lines.append(f"\n## Page {page_num}\n")
            continue

        # Convert paragraph
        if para.text.strip():
            md_lines.append(convert_paragraph_to_md(para))

        # Insert images in order
        if img_index < len(images):
            md_lines.append(f"![image]({images[img_index]})")
            img_index += 1

    # Write final Markdown
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"✔ Saved Markdown → {output_path}")


def process_all_docx(input_dir=".", output_dir="output_md"):
    os.makedirs(output_dir, exist_ok=True)

    for fname in os.listdir(input_dir):
        if fname.lower().endswith(".docx"):
            in_path = os.path.join(input_dir, fname)
            out_path = os.path.join(output_dir, fname.replace(".docx", ".md"))
            convert_docx_to_markdown(in_path, out_path)


if __name__ == "__main__":
    process_all_docx()
