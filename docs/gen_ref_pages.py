"""Generate API reference pages from the fplx package.

This script is run by mkdocs-gen-files during build.
It walks the fplx package tree and creates a .md page for each module
that contains a `::: fplx.module` directive, which mkdocstrings then
renders into full API documentation from docstrings.
"""

from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

root = Path("fplx")
ref_dir = Path("api")

for path in sorted(root.rglob("*.py")):
    module_path = path.with_suffix("")
    doc_path = path.relative_to(root).with_suffix(".md")
    full_doc_path = ref_dir / doc_path

    parts = tuple(module_path.parts)

    # Skip __pycache__
    if "__pycache__" in parts:
        continue

    # Handle __init__.py → package index page
    if parts[-1] == "__init__":
        parts = parts[:-1]
        if not parts:
            # Root __init__.py → api/index.md
            doc_path = Path("index.md")
            full_doc_path = ref_dir / doc_path
        else:
            doc_path = Path(*parts[1:]) / "index.md" if len(parts) > 1 else Path("index.md")
            full_doc_path = ref_dir / doc_path

    # Skip empty parts
    if not parts:
        continue

    # Build the Python import path (e.g., "fplx.inference.hmm")
    python_path = ".".join(parts)

    # Build a readable nav label
    if len(parts) == 1:
        nav_parts = ("fplx",)
    else:
        nav_parts = parts[1:]  # Strip "fplx" prefix for cleaner nav

    nav[nav_parts] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        fd.write(f"::: {python_path}\n")

    mkdocs_gen_files.set_edit_path(full_doc_path, path)

# Write the navigation file for literate-nav
with mkdocs_gen_files.open(ref_dir / "SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
