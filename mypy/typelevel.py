"""Type-level computation evaluation.

This module provides the evaluation functions for type-level computations
(TypeOperatorType, TypeForComprehension).

Note: Conditional types are now represented as _Cond[...] TypeOperatorType.

"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Final

from mypy.expandtype import expand_type, expand_type_by_instance
from mypy.maptype import map_instance_to_supertype
from mypy.subtypes import is_subtype
from mypy.types import (
    AnyType,
    Instance,
    LiteralType,
    NoneType,
    ProperType,
    TupleType,
    Type,
    TypedDictType,
    TypeForComprehension,
    TypeOfAny,
    TypeOperatorType,
    TypeVarType,
    UninhabitedType,
    UnionType,
    UnpackType,
    get_proper_type,
    has_type_vars,
    is_stuck_expansion,
)

from mypy.nodes import FuncDef, Var

if TYPE_CHECKING:
    from mypy.nodes import TypeInfo
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


# Registry mapping operator names (not full!) to their evaluation functions
_OPERATOR_EVALUATORS: dict[str, Callable[[TypeLevelEvaluator, TypeOperatorType], Type]] = {}


EXPANSION_ANY = AnyType(TypeOfAny.expansion_stuck)


def register_operator(
    name: str,
) -> Callable[
    [Callable[[TypeLevelEvaluator, TypeOperatorType], Type]],
    Callable[[TypeLevelEvaluator, TypeOperatorType], Type],
]:
    """Decorator to register an operator evaluator."""

    def decorator(
        func: Callable[[TypeLevelEvaluator, TypeOperatorType], Type],
    ) -> Callable[[TypeLevelEvaluator, TypeOperatorType], Type]:
        _OPERATOR_EVALUATORS[name] = func
        return func

    return decorator


def lift_over_unions(
    func: Callable[[TypeLevelEvaluator, TypeOperatorType], Type],
) -> Callable[[TypeLevelEvaluator, TypeOperatorType], Type]:
    """Decorator that lifts an operator to work over union types.

    If any argument is a union type, the operator is applied to each
    combination of union elements and the results are combined into a union.

    For example, Concat[Literal['a'] | Literal['b'], Literal['c']]
    becomes Literal['ac'] | Literal['bc'].
    """

    def wrapper(evaluator: TypeLevelEvaluator, typ: TypeOperatorType) -> Type:
        # Expand each argument, collecting union alternatives
        expanded_args: list[list[Type]] = []
        for arg in typ.args:
            proper = get_proper_type(arg)
            if isinstance(proper, UnionType):
                expanded_args.append(list(proper.items))
            else:
                expanded_args.append([arg])

        # Generate all combinations
        combinations = list(itertools.product(*expanded_args))

        # If there's only one combination, just call the function directly
        if len(combinations) == 1:
            return func(evaluator, typ)

        # Apply the operator to each combination
        results: list[Type] = []
        for combo in combinations:
            new_typ = typ.copy_modified(args=list(combo))
            result = func(evaluator, new_typ)
            # Don't include Never in unions
            # XXX: or should we get_proper_type again??
            if not (isinstance(result, ProperType) and isinstance(result, UninhabitedType)):
                results.append(result)

        if not results:
            return UninhabitedType()
        elif len(results) == 1:
            return results[0]
        else:
            return UnionType.make_union(results)

    return wrapper


class EvaluationStuck(Exception):
    pass


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

    def eval_proper(self, typ: Type) -> ProperType:
        """Main entry point: evaluate a type to its simplified form."""
        typ = get_proper_type(self.evaluate(typ))
        # A call to another expansion via an alias got stuck, reraise here
        if is_stuck_expansion(typ):
            raise EvaluationStuck
        if isinstance(typ, TypeVarType):
            raise EvaluationStuck

        return typ

    def eval_operator(self, typ: TypeOperatorType) -> Type:
        """Evaluate a type operator by dispatching to registered handler."""
        evaluator = _OPERATOR_EVALUATORS.get(typ.name)

        if evaluator is None:
            # print("NO EVALUATOR", fullname)

            # Unknown operator - return Any for now
            # In Phase 3B, unregistered operators will be handled appropriately
            return EXPANSION_ANY

        return evaluator(self, typ)

    # --- Type construction helpers ---

    def literal_bool(self, value: bool) -> LiteralType:
        """Create a Literal[True] or Literal[False] type."""
        return LiteralType(value, self.api.named_type("builtins.bool"))

    def literal_int(self, value: int) -> LiteralType:
        """Create a Literal[int] type."""
        return LiteralType(value, self.api.named_type("builtins.int"))

    def literal_str(self, value: str) -> LiteralType:
        """Create a Literal[str] type."""
        return LiteralType(value, self.api.named_type("builtins.str"))

    def tuple_type(self, items: list[Type]) -> TupleType:
        """Create a tuple type with the given items."""
        return TupleType(items, self.api.named_type("builtins.tuple"))


# --- Operator Implementations for Phase 3A ---


@register_operator("_Cond")
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


@register_operator("Iter")
def _eval_iter(evaluator: TypeLevelEvaluator, typ: TypeOperatorType) -> Type:
    """Evaluate a type-level iterator (Iter[T])."""
    if len(typ.args) != 1:
        return UninhabitedType()  # ???

    target = evaluator.eval_proper(typ.args[0])
    if isinstance(target, TupleType):
        # Check for unbounded tuple (has ..., represented by partial_fallback)
        if target.partial_fallback and not target.items:
            return UninhabitedType()
        return target
    else:
        return UninhabitedType()


@register_operator("IsSub")
def _eval_issub(evaluator: TypeLevelEvaluator, typ: TypeOperatorType) -> Type:
    """Evaluate a type-level condition (IsSub[T, Base])."""

    if len(typ.args) != 2:
        return UninhabitedType()

    lhs, rhs = typ.args

    left_proper = evaluator.eval_proper(lhs)
    right_proper = evaluator.eval_proper(rhs)

    # Handle type variables - may be undecidable
    if has_type_vars(left_proper) or has_type_vars(right_proper):
        return EXPANSION_ANY

    result = is_subtype(left_proper, right_proper)

    return evaluator.literal_bool(result)


def extract_literal_bool(typ: Type) -> bool | None:
    """Extract bool value from LiteralType."""
    typ = get_proper_type(typ)
    if isinstance(typ, LiteralType) and isinstance(typ.value, bool):
        return typ.value
    return None


def extract_literal_int(typ: Type) -> int | None:
    """Extract int value from LiteralType."""
    typ = get_proper_type(typ)
    if (
        isinstance(typ, LiteralType)
        and isinstance(typ.value, int)
        and not isinstance(typ.value, bool)
    ):
        return typ.value
    return None


def extract_literal_string(typ: Type) -> str | None:
    """Extract string value from LiteralType."""
    typ = get_proper_type(typ)
    if isinstance(typ, LiteralType) and isinstance(typ.value, str):
        return typ.value
    return None


# --- Phase 3B: Type Introspection Operators ---


@register_operator("GetArg")
@lift_over_unions
def _eval_get_arg(evaluator: TypeLevelEvaluator, typ: TypeOperatorType) -> Type:
    """Evaluate GetArg[T, Base, Idx] - get type argument at index from T as Base."""
    if len(typ.args) != 3:
        return UninhabitedType()

    target = evaluator.eval_proper(typ.args[0])
    base = evaluator.eval_proper(typ.args[1])
    idx_type = evaluator.eval_proper(typ.args[2])

    # Extract index as int
    index = extract_literal_int(idx_type)
    if index is None:
        return UninhabitedType()  # Can't evaluate without literal index

    if isinstance(target, Instance) and isinstance(base, Instance):
        args = get_type_args_for_base(target, base.type)
        if args is not None and 0 <= index < len(args):
            return args[index]
        return UninhabitedType()  # Never - invalid index or not a subtype

    return UninhabitedType()


@register_operator("GetArgs")
@lift_over_unions
def _eval_get_args(evaluator: TypeLevelEvaluator, typ: TypeOperatorType) -> Type:
    """Evaluate GetArgs[T, Base] -> tuple of all type args from T as Base."""
    if len(typ.args) != 2:
        return UninhabitedType()

    target = evaluator.eval_proper(typ.args[0])
    base = evaluator.eval_proper(typ.args[1])

    if isinstance(target, Instance) and isinstance(base, Instance):
        args = get_type_args_for_base(target, base.type)
        if args is not None:
            return evaluator.tuple_type(list(args))
        return UninhabitedType()

    return UninhabitedType()


@register_operator("FromUnion")
def _eval_from_union(evaluator: TypeLevelEvaluator, typ: TypeOperatorType) -> Type:
    """Evaluate FromUnion[T] -> tuple of union elements."""
    if len(typ.args) != 1:
        return UninhabitedType()

    target = evaluator.eval_proper(typ.args[0])

    if isinstance(target, UnionType):
        return evaluator.tuple_type(list(target.items))
    else:
        # Non-union becomes 1-tuple
        return evaluator.tuple_type([target])


@register_operator("GetAttr")
@lift_over_unions
def _eval_get_attr(evaluator: TypeLevelEvaluator, typ: TypeOperatorType) -> Type:
    """Evaluate GetAttr[T, Name] - get attribute type from T."""
    if len(typ.args) != 2:
        return UninhabitedType()

    target = evaluator.eval_proper(typ.args[0])
    name_type = evaluator.eval_proper(typ.args[1])

    name = extract_literal_string(name_type)
    if name is None:
        return UninhabitedType()

    if isinstance(target, Instance):
        node = target.type.names.get(name)
        if node is not None and node.type is not None:
            return node.type
        return UninhabitedType()

    return UninhabitedType()


# --- Phase 3B: String Operations ---


@register_operator("Slice")
@lift_over_unions
def _eval_slice(evaluator: TypeLevelEvaluator, typ: TypeOperatorType) -> Type:
    """Evaluate Slice[S, Start, End] - slice a literal string."""
    if len(typ.args) != 3:
        return UninhabitedType()

    s = extract_literal_string(evaluator.eval_proper(typ.args[0]))

    # Handle start - can be int or None
    start_type = evaluator.eval_proper(typ.args[1])
    if isinstance(start_type, NoneType):
        start: int | None = None
    else:
        start = extract_literal_int(start_type)
        if start is None:
            return UninhabitedType()

    # Handle end - can be int or None
    end_type = evaluator.eval_proper(typ.args[2])
    if isinstance(end_type, NoneType):
        end: int | None = None
    else:
        end = extract_literal_int(end_type)
        if end is None:
            return UninhabitedType()

    if s is not None:
        result = s[start:end]
        return evaluator.literal_str(result)

    return UninhabitedType()


@register_operator("Concat")
@lift_over_unions
def _eval_concat(evaluator: TypeLevelEvaluator, typ: TypeOperatorType) -> Type:
    """Evaluate Concat[S1, S2] - concatenate two literal strings."""
    if len(typ.args) != 2:
        return UninhabitedType()

    left = extract_literal_string(evaluator.eval_proper(typ.args[0]))
    right = extract_literal_string(evaluator.eval_proper(typ.args[1]))

    if left is not None and right is not None:
        return evaluator.literal_str(left + right)

    return UninhabitedType()


@register_operator("Uppercase")
@lift_over_unions
def _eval_uppercase(evaluator: TypeLevelEvaluator, typ: TypeOperatorType) -> Type:
    """Evaluate Uppercase[S] - convert literal string to uppercase."""
    if len(typ.args) != 1:
        return UninhabitedType()

    s = extract_literal_string(evaluator.eval_proper(typ.args[0]))
    if s is not None:
        return evaluator.literal_str(s.upper())

    return UninhabitedType()


@register_operator("Lowercase")
@lift_over_unions
def _eval_lowercase(evaluator: TypeLevelEvaluator, typ: TypeOperatorType) -> Type:
    """Evaluate Lowercase[S] - convert literal string to lowercase."""
    if len(typ.args) != 1:
        return UninhabitedType()

    s = extract_literal_string(evaluator.eval_proper(typ.args[0]))
    if s is not None:
        return evaluator.literal_str(s.lower())

    return UninhabitedType()


@register_operator("Capitalize")
@lift_over_unions
def _eval_capitalize(evaluator: TypeLevelEvaluator, typ: TypeOperatorType) -> Type:
    """Evaluate Capitalize[S] - capitalize first character of literal string."""
    if len(typ.args) != 1:
        return UninhabitedType()

    s = extract_literal_string(evaluator.eval_proper(typ.args[0]))
    if s is not None:
        return evaluator.literal_str(s.capitalize())

    return UninhabitedType()


@register_operator("Uncapitalize")
@lift_over_unions
def _eval_uncapitalize(evaluator: TypeLevelEvaluator, typ: TypeOperatorType) -> Type:
    """Evaluate Uncapitalize[S] - lowercase first character of literal string."""
    if len(typ.args) != 1:
        return UninhabitedType()

    s = extract_literal_string(evaluator.eval_proper(typ.args[0]))
    if s is not None:
        result = s[0].lower() + s[1:] if s else s
        return evaluator.literal_str(result)

    return UninhabitedType()


# --- Phase 3B: Object Introspection Operators ---


@register_operator("Members")
@lift_over_unions
def _eval_members(evaluator: TypeLevelEvaluator, typ: TypeOperatorType) -> Type:
    """Evaluate Members[T] -> tuple of Member types for all members of T.

    Includes methods, class variables, and instance attributes.
    """
    return _eval_members_impl(evaluator, typ, attrs_only=False)


@register_operator("Attrs")
@lift_over_unions
def _eval_attrs(evaluator: TypeLevelEvaluator, typ: TypeOperatorType) -> Type:
    """Evaluate Attrs[T] -> tuple of Member types for annotated attributes only.

    Excludes methods but includes ClassVar members.
    """
    return _eval_members_impl(evaluator, typ, attrs_only=True)


def _eval_members_impl(
    evaluator: TypeLevelEvaluator, typ: TypeOperatorType, *, attrs_only: bool
) -> Type:
    """Common implementation for Members and Attrs operators.

    Args:
        attrs_only: If True, filter to attributes only (excludes methods).
                    If False, include all members.
    """
    if len(typ.args) != 1:
        return UninhabitedType()

    target = evaluator.eval_proper(typ.args[0])

    # Get the Member TypeInfo
    member_info = evaluator.api.named_type_or_none("typing.Member")
    if member_info is None:
        return UninhabitedType()

    # Handle TypedDict
    if isinstance(target, TypedDictType):
        return _eval_typeddict_members(evaluator, target, member_info.type)

    if not isinstance(target, Instance):
        return UninhabitedType()

    members: dict[str, Type] = {}

    # Iterate through MRO in reverse (base classes first) to include inherited members
    for type_info in reversed(target.type.mro):
        # Skip types defined in stub files
        module = evaluator.api.modules.get(type_info.module_name)
        if module is not None and module.is_stub:
            continue

        for name, sym in type_info.names.items():
            # Skip private/dunder names
            if name.startswith("_"):
                continue

            if sym.type is None:
                continue

            # Skip inferred attributes (those without explicit type annotations)
            if isinstance(sym.node, Var) and sym.node.is_inferred:
                continue

            if attrs_only:
                # Attrs filters to attributes only (excludes methods).
                # Methods are FuncDef nodes; Callable-typed attributes are Var nodes.
                if isinstance(sym.node, FuncDef):
                    continue

            # Map type_info to get correct type args as seen from target
            if type_info == target.type:
                definer = target
            else:
                definer = map_instance_to_supertype(target, type_info)

            # Expand the member type to substitute type variables with actual args
            member_typ = expand_type_by_instance(sym.type, definer)

            member_type = create_member_type(
                evaluator,
                member_info.type,
                name=name,
                typ=member_typ,
                node=sym.node,
                definer=definer,
            )
            members[name] = member_type

    return evaluator.tuple_type(list(members.values()))


def _eval_typeddict_members(
    evaluator: TypeLevelEvaluator,
    target: TypedDictType,
    member_type_info: TypeInfo,
) -> Type:
    """Evaluate Members/Attrs for a TypedDict type."""
    members: list[Type] = []

    for name, item_type in target.items.items():
        # Skip private/dunder names
        if name.startswith("_"):
            continue

        # Build qualifiers for TypedDict keys
        quals: list[str] = []
        if name in target.required_keys:
            quals.append("Required")
        else:
            quals.append("NotRequired")
        if name in target.readonly_keys:
            quals.append("ReadOnly")

        # Create qualifier type
        if len(quals) == 1:
            quals_type: Type = evaluator.literal_str(quals[0])
        else:
            quals_type = UnionType.make_union(
                [evaluator.literal_str(q) for q in quals]
            )

        # For TypedDict, definer is the TypedDict's fallback instance
        definer = target.fallback

        member_type = Instance(
            member_type_info,
            [
                evaluator.literal_str(name),  # name
                item_type,  # typ
                quals_type,  # quals
                UninhabitedType(),  # init (not tracked for TypedDict)
                definer,  # definer
            ],
        )
        members.append(member_type)

    return evaluator.tuple_type(members)


def create_member_type(
    evaluator: TypeLevelEvaluator,
    member_type_info: TypeInfo,
    name: str,
    typ: Type,
    node: object,
    definer: Instance,
) -> Instance:
    """Create a Member[name, typ, quals, init, definer] instance type."""
    # Determine qualifiers
    quals: Type
    if isinstance(node, Var):
        if node.is_classvar:
            quals = evaluator.literal_str("ClassVar")
        elif node.is_final:
            quals = evaluator.literal_str("Final")
        else:
            quals = UninhabitedType()  # Never = no qualifiers
    elif isinstance(node, FuncDef):
        # Methods are class-level, so they have ClassVar qualifier
        quals = evaluator.literal_str("ClassVar")
    else:
        quals = UninhabitedType()

    # For init, we currently don't track initializer literal types
    # This would require changes to semantic analysis
    init: Type = UninhabitedType()

    return Instance(
        member_type_info,
        [
            evaluator.literal_str(name),  # name
            typ,  # typ
            quals,  # quals
            init,  # init
            definer,  # definer
        ],
    )


# --- Phase 3B: Utility Operators ---


@register_operator("Length")
@lift_over_unions
def _eval_length(evaluator: TypeLevelEvaluator, typ: TypeOperatorType) -> Type:
    """Evaluate Length[T] -> Literal[int] for tuple length."""
    if len(typ.args) != 1:
        return UninhabitedType()

    target = evaluator.eval_proper(typ.args[0])

    if isinstance(target, TupleType):
        # Check for unbounded tuple (has ..., represented by partial_fallback)
        if target.partial_fallback and not target.items:
            return NoneType()  # Unbounded tuple returns None
        return evaluator.literal_int(len(target.items))

    return UninhabitedType()


# --- Helper Functions ---


def get_type_args_for_base(instance: Instance, base_type: TypeInfo) -> tuple[Type, ...] | None:
    """Get type args when viewing instance as base class.

    Returns None if instance is not a subtype of base_type.
    """
    # Check if base_type is in the MRO. (map_instance_to_supertype
    # doesn't have a way to signal when it isn't; it just fills the
    # type with Anys)
    if base_type not in instance.type.mro:
        return None

    return map_instance_to_supertype(instance, base_type).args


# --- Public API ---


def evaluate_type_operator(typ: TypeOperatorType) -> Type:
    """Evaluate a TypeOperatorType. Called from TypeOperatorType.expand().

    Uses typelevel_ctx.api to access the semantic analyzer.
    """
    if typelevel_ctx.api is None:
        raise AssertionError("No access to semantic analyzer!")

    evaluator = TypeLevelEvaluator(typelevel_ctx.api)
    try:
        res = evaluator.eval_operator(typ)
    except EvaluationStuck:
        res = EXPANSION_ANY
    # print("EVALED!!", res)
    return res


def evaluate_comprehension(typ: TypeForComprehension) -> Type:
    """Evaluate a TypeForComprehension. Called from TypeForComprehension.expand().

    Evaluates *[Expr for var in Iter if Cond] to UnpackType(TupleType([...])).
    """
    if typelevel_ctx.api is None:
        # API not available yet - return stuck expansion marker
        return EXPANSION_ANY

    evaluator = TypeLevelEvaluator(typelevel_ctx.api)

    try:
        # Get the iterable type and expand it to a TupleType
        iter_proper = evaluator.eval_proper(typ.iter_type)
    except EvaluationStuck:
        return EXPANSION_ANY

    if not isinstance(iter_proper, TupleType):
        # Can only iterate over tuple types
        return UninhabitedType()

    # Process each item in the tuple
    result_items: list[Type] = []
    assert typ.iter_var
    for item in iter_proper.items:
        # Substitute iter_var with item in element_expr and conditions
        env = {typ.iter_var.id: item}
        substituted_expr = expand_type(typ.element_expr, env)
        substituted_conditions = [expand_type(cond, env) for cond in typ.conditions]

        # Evaluate all conditions
        try:
            all_pass = True
            for cond in substituted_conditions:
                cond_result = extract_literal_bool(evaluator.evaluate(cond))
                if cond_result is False:
                    all_pass = False
                    break
                elif cond_result is None:
                    # Undecidable condition - skip this item
                    all_pass = False
                    break

            if all_pass:
                # Include this element in the result
                result_items.append(substituted_expr)
        except EvaluationStuck:
            # Skip items that cause stuck evaluation
            continue

    return UnpackType(evaluator.tuple_type(result_items))
