# Structural Template Examples

These examples are framework-only binder templates for exercising the
path-based structural-template workflow. They are examples, not a required
framework panel.

## Contents

- `vhh/caplacizumab_vhh_7EOW.cif`: processed VHH framework-only template from
  the ESMFold2 pipeline framework set.
- `scfv/belimumab_5Y9K.cif`: processed single-chain VH-linker-VL scFv
  framework-only template from the ESMFold2 pipeline framework set.
- `binder_template_map.csv`: reusable binder template map for `run_pipeline`.
- `pairs_with_templates.csv`: minimal pair CSV showing row-local template paths.

## Use

Template paths are resolved relative to the pair CSV path or the current working
directory. The template chain ID defaults to `A`; these example files use chain
`A`.

Template-only comparison:

```bash
modal run modal_protenix_batch.py::run_pipeline \
  --pair-csv templates/examples/pairs_with_templates.csv \
  --output-dir ./results_template_only \
  --binder-mode de_novo \
  --target-msa-source none \
  --use-template true
```

Template plus target MSA comparison:

```bash
modal run modal_protenix_batch.py::run_pipeline \
  --pair-csv templates/examples/pairs_with_templates.csv \
  --output-dir ./results_template_msa \
  --binder-mode de_novo \
  --target-msa-source mmseqs \
  --use-template true \
  --mmseqs-mode colabfold \
  --mmseqs-host-url https://api.colabfold.com
```
