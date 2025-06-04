"""Generate documentation pages for built-in stages using mkdocs_gen_files."""

import mkdocs_gen_files

from MEDS_transforms.docgen import generate_stage_docs

nav = mkdocs_gen_files.Nav()
for doc in generate_stage_docs("MEDS_transforms"):
    nav[doc.stage_name] = doc.path.as_posix()
    with mkdocs_gen_files.open(doc.path, "w") as fd:
        fd.write(doc.content)
    if doc.edit_path:
        mkdocs_gen_files.set_edit_path(doc.path, doc.edit_path)

with mkdocs_gen_files.open("stages/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
