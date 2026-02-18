"""Router package – registers all FastAPI routers on the application."""

from fastapi import FastAPI


def register_routers(app: FastAPI) -> None:
    """Include all routers on the FastAPI application.

    Imports are deferred to avoid a circular import: each router module
    does ``import main``, which in turn does ``from routers import
    register_routers``.  Importing router modules at package level would
    trigger that cycle before ``register_routers`` is defined.
    """
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

    for module in (
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
    ):
        app.include_router(module.router)
