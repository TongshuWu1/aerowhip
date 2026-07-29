"""Canonical three-channel PIDNet observation and annotation schema.

Cable is one shared visual class.  Each endpoint layer contains both ends of
one physical cable, so endpoint-layer identity is also PF identity. Projected
junctions are derived from the cable-body skeleton graph rather than predicted
as a separate semantic class.
"""

PIDNET_SCHEMA_VERSION = 4
PIDNET_LABEL_MODE = "cable_with_per_cable_endpoints"
ENDPOINT_SEMANTICS = "per_cable_endpoint_sets"
ANNOTATION_SCHEMA_VERSION = 2

CABLE_CHANNEL = 0
ENDPOINT_CABLE_NAMES = ("cable1", "cable2")
ENDPOINT_CHANNELS = (1, 2)
OUTPUT_CHANNEL_COUNT = 3

# Editable layered masks map one-to-one onto the neural observation.  There are
# no cable1/cable2 body labels: cable ownership is decided by the PF.
ANNOTATION_BODY_LAYER_COUNT = 1
ANNOTATION_ENDPOINT_GROUP_COUNT = len(ENDPOINT_CABLE_NAMES)
ANNOTATION_CHANNEL_COUNT = OUTPUT_CHANNEL_COUNT


def endpoint_label_value(cable_index, _cable_count=None):
    cable_index = int(cable_index)
    if cable_index < 1 or cable_index > ANNOTATION_ENDPOINT_GROUP_COUNT:
        raise ValueError(
            f"Endpoint cable index must be in 1..{ANNOTATION_ENDPOINT_GROUP_COUNT}; got {cable_index}."
        )
    return CABLE_CHANNEL + 1 + cable_index


def max_label_value(_cable_count=None):
    return ANNOTATION_CHANNEL_COUNT


def label_bit(label):
    label = int(label)
    if label <= 0:
        return 0
    return 1 << (label - 1)


def validate_checkpoint_schema(config):
    """Return the validated per-cable endpoint count or raise on ambiguity."""

    schema_version = int(config.get("observation_schema_version", 0))
    semantics = str(config.get("endpoint_semantics", "")).strip().lower()
    endpoint_count = int(config.get("endpoint_channel_count", 0))
    label_mode = str(config.get("label_mode", "")).strip().lower()
    if schema_version != PIDNET_SCHEMA_VERSION:
        raise ValueError(
            "PIDNet checkpoint observation_schema_version is missing or incompatible; "
            f"expected {PIDNET_SCHEMA_VERSION}, got {schema_version}. Run the checkpoint schema migration tool."
        )
    if semantics != ENDPOINT_SEMANTICS:
        raise ValueError(
            f"PIDNet endpoint_semantics must be {ENDPOINT_SEMANTICS!r}; got {semantics!r}."
        )
    if endpoint_count != len(ENDPOINT_CHANNELS):
        raise ValueError(
            "PIDNet must contain endpoints_cable1 and endpoints_cable2 heads; "
            f"got {endpoint_count} endpoint channels."
        )
    if label_mode != PIDNET_LABEL_MODE:
        raise ValueError(f"PIDNet label_mode must be {PIDNET_LABEL_MODE!r}; got {label_mode!r}.")
    if int(config.get("output_channels", 0)) != OUTPUT_CHANNEL_COUNT:
        raise ValueError(
            f"PIDNet must output {OUTPUT_CHANNEL_COUNT} channels; got {config.get('output_channels')}."
        )
    return endpoint_count
