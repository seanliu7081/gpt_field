Astra scene-field PDF slides (Chinese, 17 pages)

Open: astra_field_explained_zh.pdf
Scope: existing drawer003 closed step60 / open step552 Astra reconstruction and field_planning_v2 evaluation.
No new model inference, planner run, or changes to existing experiment code/results.

Slides cover input/prompt, CSG geometry and implicit field, saved-grid reconstruction,
geometry preservation vs scene reconstruction accuracy, self-removal, joint-space planning,
truth evaluation, successful RRT example, and a native contact failure diagnosis.

Rebuild from /workspace/gpt_field:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /workspace/venvs/libero-plus/bin/python -B scene_test/astra_field_slides/build_slides.py
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /workspace/venvs/libero-plus/bin/python -B scene_test/astra_field_slides/validate_slides.py

All slide PNG previews and contact sheets are in previews/.
Evidence and source hashes: evidence_astra.json, evidence_planning.json, deck_manifest.json,
assets/evidence_figures_manifest.json. PDF validation: validation.json.
Chinese Noto Sans SC font source/license: assets/font_source.json and assets/OFL.txt.
ReportLab and PyMuPDF are scoped to vendor/; the existing virtual environment was not modified.
