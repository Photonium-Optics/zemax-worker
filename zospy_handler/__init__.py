"""ZosPy handler package – composed via mixin pattern.

Usage::

    from zospy_handler import ZosPyHandler, ZosPyError
"""

from zospy_handler._base import ZosPyHandlerBase, ZosPyError
from zospy_handler.geometry import GeometryMixin
from zospy_handler.aberrations import AberrationsMixin
from zospy_handler.performance import PerformanceMixin
from zospy_handler.ray_tracing import RayTracingMixin
from zospy_handler.ray_analysis import RayAnalysisMixin
from zospy_handler.optimization import OptimizationMixin
from zospy_handler.physical_optics import PhysicalOpticsMixin
from zospy_handler.polarization import PolarizationMixin


class ZosPyHandler(
    GeometryMixin,
    AberrationsMixin,
    PerformanceMixin,
    RayTracingMixin,
    RayAnalysisMixin,
    OptimizationMixin,
    PhysicalOpticsMixin,
    PolarizationMixin,
    ZosPyHandlerBase,
):
    """Composed ZosPy handler with all analysis mixins."""
    pass


__all__ = ["ZosPyHandler", "ZosPyError"]
