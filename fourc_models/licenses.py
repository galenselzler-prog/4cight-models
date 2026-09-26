# Copyright © 2026 4Cight Inc. All rights reserved.
# PROPRIETARY AND CONFIDENTIAL. Unauthorized copying or distribution is prohibited.
"""Licences of the open base models 4Cight's models start from, recorded in
every exported ModelSpec for IP due diligence. A base model that is not
listed here is exported as "CHECK BEFORE SHIPPING" until someone confirms it.

Each entry was checked on the model's Hugging Face card (date in comment).
Apache-2.0 and MIT allow commercial use and keeping 4Cight's fine-tuned
weights private; the app must ship their notices (THIRD_PARTY_NOTICES.md)."""

UNKNOWN = "CHECK BEFORE SHIPPING"

LICENSES = {
    "tiny": "none (randomly initialised, test only)",
    "microsoft/deberta-v3-small": "MIT",
    "microsoft/deberta-v3-xsmall": "MIT",
    "sentence-transformers/all-MiniLM-L6-v2": "Apache-2.0",
    "bert-base-uncased": "Apache-2.0",
    # M2 emotion model's base, and that model's own base (checked 2026-09-26)
    "lighteternal/wav2vec2-large-xlsr-53-greek": "Apache-2.0",
    "facebook/wav2vec2-large-xlsr-53": "Apache-2.0",
    # speaker-recognition base (checked 2026-09-26; trained on VoxCeleb, CC BY 4.0)
    "speechbrain/spkrec-ecapa-voxceleb": "Apache-2.0",
}


def license_for(base: str) -> str:
    return LICENSES.get(base, UNKNOWN)
