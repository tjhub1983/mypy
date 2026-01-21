"""Type-level computation evaluation.

This module provides the evaluation functions for type-level computations
(TypeOperatorType, TypeForComprehension).

Note: Conditional types are now represented as _Cond[...] TypeOperatorType.

Note: This is a stub implementation. The full implementation will be added
in a later phase.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Final

from mypy.types import AnyType, Type, TypeForComprehension, TypeOfAny, TypeOperatorType

if TYPE_CHECKING:
    from mypy.semanal_shared import SemanticAnalyzerCoreInterface


class TypeLevelContext:
    """Holds the context for type-level computation evaluation.

    This is a global mutable state that provides access to the semantic analyzer
    API during type operator expansion. The context is set via a context manager
    before type analysis and cleared afterward.
    """

    def __init__(self) -> None:
        self._api: SemanticAnalyzerCoreInterface | None = None

    @property
    def api(self) -> SemanticAnalyzerCoreInterface | None:
        """Get the current semantic analyzer API, or None if not in context."""
        return self._api

    @contextmanager
    def set_api(self, api: SemanticAnalyzerCoreInterface) -> Iterator[None]:
        """Context manager to set the API for type-level evaluation.

        Usage:
            with typelevel_ctx.set_api(self.api):
                # Type operators can now access the API via typelevel_ctx.api
                result = get_proper_type(some_type)
        """
        saved = self._api
        self._api = api
        try:
            yield
        finally:
            self._api = saved


# Global context instance for type-level computation
typelevel_ctx: Final = TypeLevelContext()


def evaluate_type_operator(typ: TypeOperatorType) -> Type:
    """Evaluate a TypeOperatorType. Called from TypeOperatorType.expand().

    Returns the type unchanged if evaluation is not yet possible.
    """
    # Stub implementation - return Any to avoid infinite loops in get_proper_type
    # The real implementation will:
    # 1. Check if typelevel_ctx.api is available
    # 2. If so, evaluate the type operator using the API
    # 3. If not, return the type unchanged (will be evaluated later)
    #
    # Example future implementation:
    # if typelevel_ctx.api is not None:
    #     evaluator = TypeLevelEvaluator(typelevel_ctx.api)
    #     return evaluator.eval_operator(typ)
    # return typ  # Can't evaluate yet, return unchanged
    return AnyType(TypeOfAny.special_form)


def evaluate_comprehension(typ: TypeForComprehension) -> Type:
    """Evaluate a TypeForComprehension. Called from TypeForComprehension.expand().

    Returns the type unchanged if evaluation is not yet possible.
    """
    # Stub implementation - return Any to avoid infinite loops in get_proper_type
    # The real implementation will use typelevel_ctx.api similar to evaluate_type_operator
    return AnyType(TypeOfAny.special_form)
