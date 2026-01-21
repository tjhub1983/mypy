"""Type-level computation evaluation.

This module provides the evaluation functions for type-level computations
(TypeOperatorType, TypeForComprehension).

Note: Conditional types are now represented as _Cond[...] TypeOperatorType.

Phase 3A implements: _Cond and IsSub evaluation
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Final

from mypy.subtypes import is_subtype
from mypy.types import (
    AnyType,
    LiteralType,
    Type,
    TypeForComprehension,
    TypeOfAny,
    TypeOperatorType,
    UninhabitedType,
    get_proper_type,
    has_type_vars,
)

if TYPE_CHECKING:
    from mypy.semanal_shared import SemanticAnalyzerInterface


class TypeLevelContext:
    """Holds the context for type-level computation evaluation.

    This is a global mutable state that provides access to the semantic analyzer
    API during type operator expansion. The context is set via a context manager
    before type analysis and cleared afterward.
    """

    def __init__(self) -> None:
        self._api: SemanticAnalyzerInterface | None = None

    @property
    def api(self) -> SemanticAnalyzerInterface | None:
        """Get the current semantic analyzer API, or None if not in context."""
        return self._api

    @contextmanager
    def set_api(self, api: SemanticAnalyzerInterface) -> Iterator[None]:
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


# Registry mapping operator fullnames to their evaluation functions
_OPERATOR_EVALUATORS: dict[str, Callable[[TypeLevelEvaluator, TypeOperatorType], Type]] = {}


EXPANSION_ANY = AnyType(TypeOfAny.expansion_stuck)


def register_operator(
    fullname: str,
) -> Callable[
    [Callable[[TypeLevelEvaluator, TypeOperatorType], Type]],
    Callable[[TypeLevelEvaluator, TypeOperatorType], Type],
]:
    """Decorator to register an operator evaluator."""

    def decorator(
        func: Callable[[TypeLevelEvaluator, TypeOperatorType], Type],
    ) -> Callable[[TypeLevelEvaluator, TypeOperatorType], Type]:
        _OPERATOR_EVALUATORS[fullname] = func
        return func

    return decorator


class TypeLevelEvaluator:
    """Evaluates type-level computations to concrete types.

    Phase 3A: Core conditional type evaluation (_Cond and IsSub).
    """

    def __init__(self, api: SemanticAnalyzerInterface) -> None:
        self.api = api

    def evaluate(self, typ: Type) -> Type:
        """Main entry point: evaluate a type to its simplified form."""
        if isinstance(typ, TypeOperatorType):
            return self.eval_operator(typ)
        return typ  # Already a concrete type or can't be evaluated

    def eval_operator(self, typ: TypeOperatorType) -> Type:
        """Evaluate a type operator by dispatching to registered handler."""
        fullname = typ.fullname
        evaluator = _OPERATOR_EVALUATORS.get(fullname)

        if evaluator is None:
            # print("NO EVALUATOR", fullname)

            # Unknown operator - return Any for now
            # In Phase 3B, unregistered operators will be handled appropriately
            return EXPANSION_ANY

        return evaluator(self, typ)


# --- Operator Implementations for Phase 3A ---


@register_operator("builtins._Cond")
def _eval_cond(evaluator: TypeLevelEvaluator, typ: TypeOperatorType) -> Type:
    """Evaluate _Cond[condition, TrueType, FalseType]."""

    if len(typ.args) != 3:
        return UninhabitedType()

    condition, true_type, false_type = typ.args
    result = extract_literal_bool(evaluator.evaluate(condition))

    if result is True:
        return true_type
    elif result is False:
        return false_type
    else:
        # Undecidable - return Any for now
        # In the future, we might want to keep the conditional and defer evaluation
        return EXPANSION_ANY


@register_operator("typing.IsSub")
def _eval_issub(evaluator: TypeLevelEvaluator, typ: TypeOperatorType) -> Type:
    """Evaluate a type-level condition (IsSub[T, Base])."""

    if len(typ.args) != 2:
        return UninhabitedType()

    lhs, rhs = typ.args

    left = evaluator.evaluate(lhs)
    right = evaluator.evaluate(rhs)

    # Get proper types for subtype check
    left_proper = get_proper_type(left)
    right_proper = get_proper_type(right)

    # Handle type variables - may be undecidable
    # XXX: Do I care?
    if has_type_vars(left_proper) or has_type_vars(right_proper):
        return EXPANSION_ANY

    result = is_subtype(left_proper, right_proper)

    return LiteralType(result, evaluator.api.named_type("builtins.bool"))


def extract_literal_bool(typ: Type) -> bool | None:
    """Extract int value from LiteralType."""
    typ = get_proper_type(typ)
    if isinstance(typ, LiteralType) and isinstance(typ.value, bool):
        return typ.value
    return None


# --- Public API ---


def evaluate_type_operator(typ: TypeOperatorType) -> Type:
    """Evaluate a TypeOperatorType. Called from TypeOperatorType.expand().

    Uses typelevel_ctx.api to access the semantic analyzer.
    """
    if typelevel_ctx.api is None:
        raise AssertionError("No access to semantic analyzer!")

    evaluator = TypeLevelEvaluator(typelevel_ctx.api)
    res = evaluator.eval_operator(typ)
    # print("EVALED!!", res)
    return res


def evaluate_comprehension(typ: TypeForComprehension) -> Type:
    """Evaluate a TypeForComprehension. Called from TypeForComprehension.expand().

    Returns Any for now - full implementation in Phase 3B.
    """
    # Stub implementation - return Any to avoid infinite loops in get_proper_type
    return EXPANSION_ANY
