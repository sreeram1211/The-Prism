"""
Main API router — aggregates all endpoint sub-routers.

Import order here controls the order of sections in the OpenAPI docs.
"""

from fastapi import APIRouter

from .endpoints import agent, compare, dashboard, generate, monitor, resolver, scan

api_router = APIRouter()

api_router.include_router(resolver.router)
api_router.include_router(scan.router)
api_router.include_router(generate.router)
api_router.include_router(monitor.router)
api_router.include_router(agent.router)
api_router.include_router(compare.router)
api_router.include_router(dashboard.router)
