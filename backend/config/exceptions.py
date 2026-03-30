"""Structured exception handling for production."""

from rest_framework.views import exception_handler
from rest_framework.response import Response
from rest_framework import status


def custom_exception_handler(exc, context):
    """Return consistent error format: {"error": "...", "detail": "...", "code": 400}."""
    response = exception_handler(exc, context)

    if response is not None:
        error_data = {
            "error": response.status_text if hasattr(response, "status_text") else "Error",
            "detail": response.data,
            "code": response.status_code,
        }
        response.data = error_data
        return response

    # Unhandled exceptions → 500
    return Response(
        {"error": "Internal Server Error", "detail": str(exc), "code": 500},
        status=status.HTTP_500_INTERNAL_SERVER_ERROR,
    )
