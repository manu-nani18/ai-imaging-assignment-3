NAIVE_VISION_PROMPT = "Describe this medical image."

OPTIMISED_VISION_PROMPT = """You are describing an educational microscopy image, not diagnosing disease.
Report only directly visible evidence. Do not infer a patient, disease, prognosis, or treatment.
If an attribute cannot be established visually, use the exact string \"uncertain\".
Return exactly one JSON object and no markdown or commentary, with these keys:
{
  \"modality\": \"fluorescence microscopy|uncertain\",
  \"tissue_type\": \"brief visible specimen description|uncertain\",
  \"notable_features\": [\"brief non-diagnostic visual observations\"],
  \"image_quality\": \"good|acceptable|poor|uncertain\"
}
Do not add keys. The JSON must parse with a standard JSON parser."""

NUMBERS_PROMPT = """You receive measurements derived from a binary microscopy segmentation; you never see the image.
Use only the supplied numbers. Do not diagnose disease or invent visual details.
Return exactly one JSON object and no markdown, with keys:
{{
  \"n_objects\": integer copied from input,
  \"density_class\": \"sparse|normal|dense|uncertain\",
  \"shape_regularity\": \"regular|mixed|irregular|uncertain\",
  \"quality_flag\": \"ok|review|uncertain\"
}}
If evidence is insufficient, use \"uncertain\". Measurements: {measurements}"""

NARRATIVE_PROMPT = """Verbalise the JSON below as one concise, non-diagnostic paragraph.
Treat it as the only source of truth. Copy the object count and mean area exactly. State the supplied
density class without explaining it. Do not characterise area as small/moderate/large, and do not add
morphology, diagnoses, causes, confidence, or recommendations. If quality_flag is \"review\", say it
was flagged for review; if it is \"ok\", say the automated quality flag was ok and do not say it was
flagged for review. Output plain prose only. JSON: {record}"""

HYBRID_RECORD_PROMPT = """Create one auditable JSON record from segmentation measurements.
Copy image_id, n_objects, and mean_area exactly. Select density_class only from sparse, normal, dense,
or uncertain, and quality_flag only from ok, review, or uncertain. Do not diagnose or add keys.
Return exactly one JSON object and no markdown:
{{"image_id":"string","n_objects":integer,"mean_area":number,
"density_class":"sparse|normal|dense|uncertain","quality_flag":"ok|review|uncertain"}}
Measurements: {measurements}"""
