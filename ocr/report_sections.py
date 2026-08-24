"""
Parse structured sections from SmartDoseAI-style lab PDFs (diagnosis line + bullet labs).
"""
import re


def extract_stated_diagnosis_from_text(text):
    """Reads 'Final Specific Diagnosis' block when present."""
    if not text:
        return None
    m = re.search(
        r"Final\s+Specific\s+Diagnosis\s*\n+\s*(.+?)"
        r"(?=\n\s*(?:Laboratory\s+Investigations|Key\s|Clinical\s|Patient\s)|\Z)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if not m:
        return None
    line = re.sub(r"\s+", " ", m.group(1)).strip()
    return line or None


def extract_laboratory_investigations(text):
    """
    Bullet lines after 'Laboratory Investigations'.
    Returns dict with snake-ish keys; numeric prefix extracted when possible.
    """
    if not text:
        return {}
    m = re.search(
        r"Laboratory\s+Investigations\s*\n+(.*?)(?=\n\s*\n|\nClinical\s|\Z)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if not m:
        return {}
    block = m.group(1)
    out = {}
    for raw in block.splitlines():
        line = raw.strip()
        if not line:
            continue
        b = re.match(r"^[-•]\s*(.+?)\s*:\s*(.+)$", line)
        if not b:
            continue
        label = b.group(1).strip()
        val_raw = b.group(2).strip()
        key = re.sub(r"[^\w]+", "_", label).strip("_")
        if not key:
            continue
        num = re.search(r"(?:<\s*)?([\d.]+)", val_raw)
        if num:
            try:
                out[key] = float(num.group(1))
                continue
            except ValueError:
                pass
        out[key] = val_raw
    return out


# Canonical display keys -> alternate keys emitted by PDF bullets or regex.
_HEART_ALIAS_GROUPS = (
    ("Heart_Rate", ("Resting_Heart_Rate", "Resting_HR")),
    ("Troponin", ("High_sensitivity_Troponin_I", "Troponin_I", "hs_Troponin_I")),
    ("NT-proBNP", ("NT_proBNP", "NTproBNP", "N_terminal_proBNP")),
    ("Potassium", ("Serum_Potassium",)),
    ("Magnesium", ("Serum_Magnesium",)),
)


def canonicalize_lab_parameter_keys(parameters):
    """
    Merge duplicate physiology keys (e.g. Serum Magnesium vs Magnesium) into one card.
    Returns a new dict; unknown keys are preserved.
    """
    if not parameters:
        return {}
    out = dict(parameters)
    for canonical, aliases in _HEART_ALIAS_GROUPS:
        val = out.get(canonical)
        if val is None:
            for a in aliases:
                if a in out and out[a] is not None:
                    val = out[a]
                    break
        if val is not None:
            out[canonical] = val
        for a in aliases:
            out.pop(a, None)
    return out


def harmonize_heart_aliases(parameters):
    """Map bulletin-board keys onto extractor canonical keys; drop aliases (fixes duplicate cards)."""
    if not parameters:
        return parameters
    merged = canonicalize_lab_parameter_keys(parameters)
    parameters.clear()
    parameters.update(merged)
    return parameters
