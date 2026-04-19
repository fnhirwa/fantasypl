"""Generate API reference pages from the fplx package.

This script is run by mkdocs-gen-files during build.
It walks the fplx package tree and creates a .md page for each module
that contains a `::: fplx.module` directive, which mkdocstrings then
renders into full API documentation from docstrings.
"""

from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

src = Path(".")
ref_dir = Path("api")
subpackages = []

for path in sorted(Path("fplx").rglob("*.py")):
    if "__pycache__" in path.parts:
        continue

    module_path = path.relative_to(src).with_suffix("")
    doc_path = path.relative_to(src).with_suffix(".md")
    full_doc_path = ref_dir / doc_path

    parts = tuple(module_path.parts)

    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")
    elif parts[-1] == "__main__" or parts[-1].startswith("_"):
        continue

    if not parts:
        continue

    # Collect direct subpackages for the landing page
    if len(parts) == 2 and (Path("fplx") / parts[1] / "__init__.py").exists():
        subpackages.append(parts[1])

    nav[parts] = doc_path.as_posix()

    # For the root fplx package, generate a landing page instead of ::: fplx
    if parts == ("fplx",):
        continue

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        ident = ".".join(parts)
        print(f"::: {ident}", file=fd)

    mkdocs_gen_files.set_edit_path(full_doc_path, path)

# Write the root API landing page
with mkdocs_gen_files.open(ref_dir / "fplx" / "index.md", "w") as fd:
    print("# fplx", file=fd)
    print("", file=fd)
    print("::: fplx", file=fd)
    print("    options:", file=fd)
    print("      show_submodules: true", file=fd)
    print("", file=fd)
    print("## Subpackages", file=fd)
    print("", file=fd)
    for pkg in sorted(set(subpackages)):
        label = f"fplx.{pkg}"
        print(f"- [`{label}`]({pkg}/index.md)", file=fd)

# Write the navigation file for literate-nav
with mkdocs_gen_files.open(ref_dir / "SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
