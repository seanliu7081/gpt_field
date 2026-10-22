SpeedUp v1 — framework, technical methods and recorded results
Chinese PDF, 24 pages

Open: speedup_v1_framework_methods_results_zh.pdf

Pages 2–6: architecture, exact inputs, model contract, skills and event loop.
Pages 7–17: calibration, pointing, visual Jacobian, contact-driven grasping,
waypoint field, analytic local planner, PathTracker and transport/place feedback.
Pages 18–23: task outcomes, recorded frames, timing, historical reference and limits.
Page 24: evidence index.

Primary evidence: frozen drawer003 method (prompt5), bowl002 result (prompt2).
The v1 field is a local waypoint distance field; it is not the later Astra CSG scene field.
Raw LIBERO success, contact monitoring, skill outcomes and task completion are separated.
No new inference or simulator task was run; original code and results are untouched.

Rebuild from /workspace/gpt_field:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /workspace/venvs/libero-plus/bin/python -B presentations/speedup_v1/build_slides.py
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /workspace/venvs/libero-plus/bin/python -B presentations/speedup_v1/validate_slides.py

The builder reuses scoped ReportLab/PyMuPDF from scene_test/astra_field_slides/vendor.
Fonts are embedded in the PDF; font license and provenance are under assets/.
Previews: previews/slide_*.png and contact_sheet_*.jpg.
Evidence: evidence_framework.json, evidence_methods.json, evidence_results.json.
Source integrity: deck_manifest.json (62 hashed sources).
Validation: validation.json. Exact model-image and video-frame provenance is retained.
