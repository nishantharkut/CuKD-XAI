"""Final, additive hardware-in-the-loop campaign tooling.

This package does not replace the historical USB or Wi-Fi HIL paths.  It binds
the final seed-42 exports to a versioned campaign, generated firmware bundles,
stage attempts, and completion evidence.
"""

from .contracts import (
    CAMPAIGN_PROTOCOL_ID,
    FINAL_STAGES,
    MODEL_KEYS,
    FinalExportIdentity,
    build_campaign_contract,
    validate_campaign_contract,
    validate_final_export,
)

__all__ = [
    "CAMPAIGN_PROTOCOL_ID",
    "FINAL_STAGES",
    "MODEL_KEYS",
    "FinalExportIdentity",
    "build_campaign_contract",
    "validate_campaign_contract",
    "validate_final_export",
]

