"""Generate documentation pages for built-in stages using mkdocs_gen_files."""

from pathlib import Path

import mkdocs_gen_files

from MEDS_transforms.stages.docgen import generate_stage_docs

nav = mkdocs_gen_files.Nav()

for doc in generate_stage_docs("MEDS_transforms"):
    nav_key = Path(doc.path)
    doc_path = nav_key / "index.md"
    full_doc_path = Path("stages") / doc_path

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        fd.write(doc.content)

    nav[nav_key.parts] = doc_path.as_posix()

    if doc.edit_path:
        mkdocs_gen_files.set_edit_path(doc.path, doc.edit_path)

with mkdocs_gen_files.open("stages/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
