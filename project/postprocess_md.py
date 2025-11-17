import os
import re

INPUT_MD_DIR = "out_md"
OUTPUT_MD_DIR = "out_md_clean"

os.makedirs(OUTPUT_MD_DIR, exist_ok=True)

# ---------------------------
# HEURISTIC RULES
# ---------------------------

def is_heading_candidate(line):
    """
    Determines if a line looks like a heading.
    We do NOT modify '## Page X'.
    """
    stripped = line.strip()

    # Already a heading → keep
    if stripped.startswith("#"):
        return True

    # Skip empty lines
    if not stripped:
        return False

    # Skip page markers
    if re.match(r"^Page\s+\d+$", stripped, re.IGNORECASE):
        return False

    # ALL CAPS short line (2–8 words)
    if stripped.isupper() and 1 <= len(stripped.split()) <= 8:
        return True

    # Ends with colon → likely a heading
    if stripped.endswith(":") and len(stripped.split()) <= 10:
        return True

    # Very short lines (< 6 words) that look like titles
    if len(stripped.split()) <= 6 and stripped[0].isupper():
        return True

    return False


def fix_headings(lines):
    cleaned = []
    for line in lines:
        stripped = line.strip()

        # Keep page markers exactly as-is
        if re.match(r"^Page\s+\d+$", stripped, re.IGNORECASE):
            cleaned.append(f"## {stripped}")
            continue

        # Promote heading candidates
        if is_heading_candidate(stripped):
            cleaned.append(f"### {stripped}")
        else:
            cleaned.append(line)

    return cleaned


def add_toc(lines):
    """Add a simple TOC at the top listing all headings."""
    toc = ["# Table of Contents\n"]

    for line in lines:
        if line.startswith("### ") and "Page " not in line:
            title = line.replace("### ", "").strip()
            anchor = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
            toc.append(f"- [{title}](#{anchor})")

    toc.append("\n")
    return toc + lines


def process_markdown(md_path, out_path):
    with open(md_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Step 1: fix headings
    lines = fix_headings(lines)

    # Step 2: add table of contents
    lines = add_toc(lines)

    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def run():
    for fname in os.listdir(INPUT_MD_DIR):
        if not fname.endswith(".md"):
            continue

        input_path = os.path.join(INPUT_MD_DIR, fname)
        output_path = os.path.join(OUTPUT_MD_DIR, fname)

        print(f"→ Cleaning {fname}")
        process_markdown(input_path, output_path)

    print("\n✅ Markdown post-processing complete. Clean files in out_md_clean/")


if __name__ == "__main__":
    run()
