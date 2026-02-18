"""Router package – registers all FastAPI routers on the application."""

from fastapi import FastAPI

from routers import (
    aberrations,
    core,
    diagnostics,
    field_analysis,
    geometry,
    merit,
    optimization,
    performance,
    physical_optics,
    polarization,
)

_ALL_ROUTERS = [
    core,
    aberrations,
    performance,
    field_analysis,
    geometry,
    merit,
    optimization,
    physical_optics,
    polarization,
    diagnostics,
]


def register_routers(app: FastAPI) -> None:
    """Include all routers on the FastAPI application."""
    for module in _ALL_ROUTERS:
        app.include_router(module.router)
