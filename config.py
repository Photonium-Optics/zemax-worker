"""
Zemax Worker Configuration

Centralized constants for the worker process and ZosPy handler.
"""

import os

# =============================================================================
# Server configuration
# =============================================================================

# Error messages
NOT_CONNECTED_ERROR = "OpticStudio not connected"

# Default server configuration
DEFAULT_PORT = 8787
DEFAULT_HOST = "0.0.0.0"

# API key for authentication (optional but recommended)
ZEMAX_API_KEY = os.getenv("ZEMAX_API_KEY", None)

# Number of uvicorn workers behind this URL. Read from WEB_CONCURRENCY (set by
# __main__ or the operator). The analysis service reads this via /health to size
# its task queue. Defaults to 1 if unset.
WORKER_COUNT = int(os.getenv("WEB_CONCURRENCY", "1"))

# =============================================================================
# Reconnect backoff
# =============================================================================

_RECONNECT_BACKOFF_BASE = 3.0   # seconds
_RECONNECT_BACKOFF_MAX = 60.0   # seconds
_RECONNECT_COM_RELEASE_DELAY = 2.0  # seconds to wait after close() for COM cleanup

# =============================================================================
# ZosPy handler constants
# =============================================================================

# Analysis settings
DEFAULT_NUM_CROSS_SECTION_RAYS = 11

# ZOSAPI FieldType enum name -> (internal field_type, unit)
FIELD_TYPE_MAP = {
    "Angle": ("object_angle", "deg"),
    "TheodoliteAngle": ("object_angle", "deg"),
    "ObjectHeight": ("object_height", "mm"),
    "ParaxialImageHeight": ("image_height", "mm"),
    "RealImageHeight": ("image_height", "mm"),
}

# Image export settings
CROSS_SECTION_IMAGE_SIZE = (1200, 800)
CROSS_SECTION_TEMP_FILENAME = "zemax_cross_section.png"
SEIDEL_TEMP_FILENAME = "seidel_native.txt"

# OpticStudio version requirements
MIN_IMAGE_EXPORT_VERSION = (24, 1, 0)

# Ray trace error codes
RAY_ERROR_CODES = {
    0: "OK",
    1: "MISS",
    2: "TIR",
    3: "REVERSED",
    4: "VIGNETTE",
}

# Aperture types where ApertureValue directly returns the F/#
# Note: FloatByStopSize is NOT included — its ApertureValue is the stop
# semi-diameter scaling factor (often 0), not the F/#.
FNO_APERTURE_TYPES = ["ImageSpaceFNum", "ParaxialWorkingFNum"]

# =============================================================================
# Logging/output formatting
# =============================================================================

# Maximum characters per raw output log message
_RAW_LOG_MAX_CHARS = 4000

# Fields to summarize (show first N elements + count)
_ARRAY_SUMMARY_FIELDS = {
    "spot_data", "spot_rays", "data", "evaluated_rows",
    "zernike_coefficients", "raw_rays", "generated_rows",
    "surface_semi_diameters", "surfaces", "frequency",
    "diffraction_limit", "tangential", "sagittal",
}
_ARRAY_SUMMARY_MAX = 5

# Fields containing binary/large base64 data to skip
_BINARY_FIELDS = {"image", "zmx_content", "modified_zmx_content"}

# Text fields to truncate
_TEXT_TRUNCATE_FIELDS = {"seidel_text": 200}
