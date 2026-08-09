"""Custom model wrappers for the TabBench-LLM pipeline.

AutoGluon model classes that aren't in AutoGluon's own registry live here, one
module per model, so they're grouped and discoverable rather than scattered at the
package root. (Mirrors the original RamanBench ``models/`` package; flattened — no
``custom/`` sublevel.)

Currently:

- :class:`~tabbench_llm.models.tabpfn_wide.TabPFNWideModel` — TabPFN-Wide
  (wide-dataset TabPFN variant) wrapped for AutoGluon, selectable via the model key
  ``"TABPFN-WIDE"``. Classification only.
- :class:`~tabbench_llm.models.tabfm.TabFMModel` — TabFM (Google Research's zero-shot
  tabular foundation model) wrapped for AutoGluon, selectable via the model key
  ``"TABFM"``. Supports both classification and regression.

Import the wrapper from its submodule (``from tabbench_llm.models.tabpfn_wide import
TabPFNWideModel``) so AutoGluon is only imported when the model is actually used.
"""
