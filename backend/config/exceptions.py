"""Structured exception handling.

All API responses follow format:
{
    "success": false,
    "error": {
        "code": "VALIDATION_ERROR",
        "message": "Field 'email' is required.",
        "status": 400
    }
}
"""

from rest_framework import status
from rest_framework.exceptions import (
    AuthenticationFailed,
    NotAuthenticated,
    NotFound,
    ParseError,
    PermissionDenied,
    ValidationError,
)
from rest_framework.response import Response
from rest_framework.views import exception_handler

# Error code mapping
_ERROR_CODES = {
    400: "BAD_REQUEST",
    401: "UNAUTHORIZED",
    403: "FORBIDDEN",
    404: "NOT_FOUND",
    405: "METHOD_NOT_ALLOWED",
    422: "UNPROCESSABLE_ENTITY",
    429: "RATE_LIMITED",
    500: "INTERNAL_ERROR",
}


def custom_exception_handler(exc, context):
    """Return consistent error format for all exceptions."""
    response = exception_handler(exc, context)

    if response is not None:
        error_code = _ERROR_CODES.get(response.status_code, "ERROR")

        # Extract message
        if isinstance(response.data, dict):
            if "detail" in response.data:
                message = str(response.data["detail"])
            else:
                # Validation errors: flatten field errors
                messages = []
                for field, errors in response.data.items():
                    if isinstance(errors, list):
                        for err in errors:
                            messages.append(f"{field}: {err}")
                    else:
                        messages.append(f"{field}: {errors}")
                message = "; ".join(messages) if messages else str(response.data)
        elif isinstance(response.data, list):
            message = "; ".join(str(e) for e in response.data)
        else:
            message = str(response.data)

        response.data = {
            "success": False,
            "error": {
                "code": error_code,
                "message": message,
                "status": response.status_code,
            },
        }
        return response

    # Unhandled exceptions → 500
    return Response(
        {
            "success": False,
            "error": {
                "code": "INTERNAL_ERROR",
                "message": str(exc) if str(exc) else "An unexpected error occurred.",
                "status": 500,
            },
        },
        status=status.HTTP_500_INTERNAL_SERVER_ERROR,
    )


class AppError(Exception):
    """Base application error — raise in views/services for clean error responses."""

    def __init__(self, message: str, code: str = "BAD_REQUEST", status_code: int = 400):
        self.message = message
        self.code = code
        self.status_code = status_code
        super().__init__(message)
