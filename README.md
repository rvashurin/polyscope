# PolyScope

PolyScope is a local Streamlit application to interactively explore saved manager archives produced by the LM-Polygraph benchmarking library for uncertainty quantification in LLMs.

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
streamlit run polyscope.py
```
Upload your `.pt`/`.pth`/`.man` archive when prompted and use the sidebar controls to inspect meta information, filter the data table, and plot rejection curves.

## Archive Format
The input archive should be a PyTorch-saved dict with the following structure:

- **estimations**: dict mapping `(sequence, method_name)` → list of uncertainties per data point  
- **stats**: dict mapping stat names → list of stat values (numeric or text) per data point  
- **gen_metrics**: dict mapping `(sequence, metric_name)` → list of quality metrics per data point  
-- Other top-level keys (e.g. `model_name`, `run_timestamp`) are treated as meta fields.

## Column Acronyms

Optionally define a `column_acronyms.json` file in the project root to map full column names to acronyms.
The app will auto-load this JSON and rename columns accordingly. Example `column_acronyms.json`:

```json
{
    "MonteCarloSequenceEntropy": "MCSE",
    "MonteCarloNormalizedSequenceEntropy": "MCNSE",
}
```

## Rejection Curve

The rejection plot now displays two curves:

- **Uncertainty-based**: removes data points in descending order of the selected uncertainty method.
- **Oracle**: removes data points in ascending order of the selected quality metric (optimal removal of the worst examples).

The Y-axis is scaled to start just below the minimum metric value for better visual focus.
