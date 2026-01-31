"""Type-level computation evaluation.

This module provides the evaluation functions for type-level computations
(TypeOperatorType, TypeForComprehension).

Note: Conditional types are now represented as _Cond[...] TypeOperatorType.

"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Final

from mypy.expandtype import expand_type, expand_type_by_instance
from mypy.maptype import map_instance_to_supertype
from mypy.nodes import Context, FuncDef, Var
from mypy.subtypes import is_subtype
from mypy.types import (
    AnyType,
    ComputedType,
    Instance,
    LiteralType,
    NoneType,
    ProperType,
    TupleType,
    Type,
    TypeAliasType,
    TypedDictType,
    TypeForComprehension,
    TypeOfAny,
    TypeOperatorType,
    TypeVarLikeType,
    UnboundType,
    UninhabitedType,
    UnionType,
    UnpackType,
    get_proper_type,
    has_type_vars,
    is_stuck_expansion,
)

if TYPE_CHECKING:
    from mypy.nodes import TypeInfo
    from mypy.semanal_shared import SemanticAnalyzerInterface


MAX_DEPTH = 100


class TypeLevelContext:
    """Holds the context for type-level computation evaluation.

    This is a global mutable state that provides access to the semantic analyzer
    API during type operator expansion. The context is set via a context manager
    before type analysis and cleared afterward.
    """

    def __init__(self) -> None:
        self._api: SemanticAnalyzerInterface | None = None
        # Make an evaluator part of this state also, so that we can
        # maintain a depth tracker and an outer error message context.
        #
        # XXX: but maybe we should always thread the evaluator back
        # ourselves or something instead?
        self._evaluator: TypeLevelEvaluator | None = None

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

EXPANSION_OVERFLOW = AnyType(TypeOfAny.from_error)


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

        return UnionType.make_union(results)

    return wrapper


class EvaluationStuck(Exception):
    pass


class EvaluationOverflow(Exception):
    pass


class TypeLevelEvaluator:
    """Evaluates type-level computations to concrete types."""

    def __init__(self, api: SemanticAnalyzerInterface, ctx: Context | None) -> None:
        self.api = api
        self.ctx = ctx
        self.depth = 0

        self.cache: dict[Type, Type] = {}

    def evaluate(self, typ: Type) -> Type:
        """Main entry point: evaluate a type to its simplified form."""

        if typ in self.cache:
            return self.cache[typ]

        if self.depth >= MAX_DEPTH:
            ctx = self.ctx or typ
            # Use serious=True to bypass in_checked_function() check which requires
            # self.options to be set on the SemanticAnalyzer
            self.api.fail("Type expansion is too deep; producing Any", ctx, serious=True)
            raise EvaluationOverflow()

        try:
            self.depth += 1
            if isinstance(typ, TypeOperatorType):
                rtyp = self.eval_operator(typ)
            elif isinstance(typ, TypeForComprehension):
                rtyp = evaluate_comprehension(self, typ)
            else:
                rtyp = typ  # Already a concrete type or can't be evaluated

            self.cache[typ] = rtyp

            return rtyp
        finally:
            self.depth -= 1

    def eval_proper(self, typ: Type) -> ProperType:
        """Main entry point: evaluate a type to its simplified form."""
        typ = get_proper_type(self.evaluate(typ))
        # A call to another expansion via an alias got stuck, reraise here
        if is_stuck_expansion(typ):
            raise EvaluationStuck
        if isinstance(typ, (TypeVarLikeType, UnboundType, ComputedType)):
            raise EvaluationStuck
        if isinstance(typ, UnpackType) and isinstance(typ.type, TypeVarLikeType):
            raise EvaluationStuck

        return typ

    def eval_operator(self, typ: TypeOperatorType) -> Type:
        """Evaluate a type operator by dispatching to registered handler."""
        evaluator = _OPERATOR_EVALUATORS.get(typ.type.name)

        if evaluator is None:
            # print("NO EVALUATOR", fullname)

            # Unknown operator - return Any for now
            return EXPANSION_ANY

        return evaluator(self, typ)

    # --- Type construction helpers ---

    def literal_bool(self, value: bool) -> LiteralType:
        """Create a Literal[True] or Literal[False] type."""
        return LiteralType(value, self.api.named_type("builtins.bool"))

    def literal_int(self, value: int) -> LiteralType:
        """Create a Literal[int] type."""
        return LiteralType(value, self.api.named_type("builtins.int"))

    def flatten_args(self, args: list[Type]) -> list[Type]:
        """Flatten type arguments, evaluating and unpacking as needed.

        Handles UnpackType from comprehensions by expanding the inner TupleType.
        """
        flat_args: list[Type] = []
        for arg in args:
            evaluated = self.eval_proper(arg)
            if isinstance(evaluated, UnpackType):
                inner = get_proper_type(evaluated.type)
                if isinstance(inner, TupleType):
                    flat_args.extend(inner.items)
                else:
                    flat_args.append(evaluated)
            else:
                flat_args.append(evaluated)
        return flat_args

    def literal_str(self, value: str) -> LiteralType:
        """Create a Literal[str] type."""
        return LiteralType(value, self.api.named_type("builtins.str"))

    def tuple_type(self, items: list[Type]) -> TupleType:
        """Create a tuple type with the given items."""
        return TupleType(items, self.api.named_type("builtins.tuple"))


def _call_by_value(evaluator: TypeLevelEvaluator, typ: Type) -> Type:
    """Make sure alias arguments are evaluated before expansion.

    Currently this is used in conditional bodies, which should protect
    any recursive uses, to make sure that arguments to potentially
    recursive aliases get evaluated before substituted in, to make
    sure that they don't grow without bound.

    This shouldn't be necessary for correctness, but can be important
    for performance.

    We should *maybe* do it in more places! Possibly everywhere?  Or
    maybe we should do it *never* and just do a better job of caching.
    """
    if isinstance(typ, TypeAliasType):
        typ = typ.copy_modified(
            args=[get_proper_type(_call_by_value(evaluator, st)) for st in typ.args]
        )

    # Evaluate recursively instead of letting it get handled in the
    # get_proper_type loop to help maintain better error contexts.
    return evaluator.eval_proper(typ)


@register_operator("_Cond")
def _eval_cond(evaluator: TypeLevelEvaluator, typ: TypeOperatorType) -> Type:
    """Evaluate _Cond[condition, TrueType, FalseType]."""

    if len(typ.args) != 3:
        return UninhabitedType()

    condition, true_type, false_type = typ.args
    result = extract_literal_bool(evaluator.eval_proper(condition))

    if result is True:
        return _call_by_value(evaluator, true_type)
    elif result is False:
        return _call_by_value(evaluator, false_type)
    else:
        # Undecidable - return Any for now
        # In the future, we might want to keep the conditional and defer evaluation
        return EXPANSION_ANY


@register_operator("_And")
def _eval_and(evaluator: TypeLevelEvaluator, typ: TypeOperatorType) -> Type:
    """Evaluate _And[cond1, cond2] - logical AND of type booleans."""
    if len(typ.args) != 2:
        return UninhabitedType()

    left = extract_literal_bool(evaluator.eval_proper(typ.args[0]))
    if left is False:
        # Short-circuit: False and X = False
        return evaluator.literal_bool(False)
    if left is None:
        return UninhabitedType()

    right = extract_literal_bool(evaluator.eval_proper(typ.args[1]))
    if right is None:
        return UninhabitedType()

    return evaluator.literal_bool(right)


@register_operator("_Or")
def _eval_or(evaluator: TypeLevelEvaluator, typ: TypeOperatorType) -> Type:
    """Evaluate _Or[cond1, cond2] - logical OR of type booleans."""
    if len(typ.args) != 2:
        return UninhabitedType()

    left = extract_literal_bool(evaluator.eval_proper(typ.args[0]))
    if left is True:
        # Short-circuit: True or X = True
        return evaluator.literal_bool(True)
    if left is None:
        return UninhabitedType()

    right = extract_literal_bool(evaluator.eval_proper(typ.args[1]))
    if right is None:
        return UninhabitedType()

    return evaluator.literal_bool(right)


@register_operator("_Not")
def _eval_not(evaluator: TypeLevelEvaluator, typ: TypeOperatorType) -> Type:
    """Evaluate _Not[cond] - logical NOT of a type boolean."""
    if len(typ.args) != 1:
        return UninhabitedType()

    result = extract_literal_bool(evaluator.eval_proper(typ.args[0]))
    if result is None:
        return UninhabitedType()

    return evaluator.literal_bool(not result)


@register_operator("Iter")
def _eval_iter(evaluator: TypeLevelEvaluator, typ: TypeOperatorType) -> Type:
    """Evaluate a type-level iterator (Iter[T])."""
    if len(typ.args) != 1:
        return UninhabitedType()  # ???

    target = evaluator.eval_proper(typ.args[0])
    if isinstance(target, TupleType):
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


@register_operator("Matches")
def _eval_matches(evaluator: TypeLevelEvaluator, typ: TypeOperatorType) -> Type:
    """Evaluate Matches[T, S] - check if T and S are equivalent types.

    Returns Literal[True] if T is a subtype of S AND S is a subtype of T.
    Equivalent to: IsSub[T, S] and IsSub[S, T]
    """
    if len(typ.args) != 2:
        return UninhabitedType()

    lhs, rhs = typ.args

    left_proper = evaluator.eval_proper(lhs)
    right_proper = evaluator.eval_proper(rhs)

    # Handle type variables - may be undecidable
    if has_type_vars(left_proper) or has_type_vars(right_proper):
        return EXPANSION_ANY

    # Both directions must hold for type equivalence
    result = is_subtype(left_proper, right_proper) and is_subtype(right_proper, left_proper)

    return evaluator.literal_bool(result)


@register_operator("Bool")
def _eval_bool(evaluator: TypeLevelEvaluator, typ: TypeOperatorType) -> Type:
    """Evaluate Bool[T] - check if T contains Literal[True].

    Returns Literal[True] if T is Literal[True] or a union containing Literal[True].
    Equivalent to: IsSub[Literal[True], T] and not IsSub[T, Never]
    """
    if len(typ.args) != 1:
        return UninhabitedType()

    arg_proper = evaluator.eval_proper(typ.args[0])

    # Check if Literal[True] is a subtype of arg (i.e., arg contains True)
    # and arg is not Never
    literal_true = evaluator.literal_bool(True)
    contains_true = is_subtype(literal_true, arg_proper)
    is_never = isinstance(arg_proper, UninhabitedType)

    return evaluator.literal_bool(contains_true and not is_never)


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


def _get_args(evaluator: TypeLevelEvaluator, target: Type, base: Type) -> Sequence[Type] | None:
    target = evaluator.eval_proper(target)
    base = evaluator.eval_proper(base)

    # TODO: Other cases
    if isinstance(target, Instance) and isinstance(base, Instance):
        return get_type_args_for_base(target, base.type)

    if (
        isinstance(target, TupleType)
        and isinstance(base, Instance)
        # XXX: Do a real check
        and target.partial_fallback == base
    ):
        return target.items

    return None


@register_operator("GetArg")
@lift_over_unions
def _eval_get_arg(evaluator: TypeLevelEvaluator, typ: TypeOperatorType) -> Type:
    """Evaluate GetArg[T, Base, Idx] - get type argument at index from T as Base."""
    if len(typ.args) != 3:
        return UninhabitedType()

    args = _get_args(evaluator, typ.args[0], typ.args[1])

    if args is None:
        return UninhabitedType()

    # Extract index as int
    index = extract_literal_int(evaluator.eval_proper(typ.args[2]))
    if index is None:
        return UninhabitedType()  # Can't evaluate without literal index

    if index < 0:
        index += len(args)
    if 0 <= index < len(args):
        return args[index]

    return UninhabitedType()


@register_operator("GetArgs")
@lift_over_unions
def _eval_get_args(evaluator: TypeLevelEvaluator, typ: TypeOperatorType) -> Type:
    """Evaluate GetArgs[T, Base] -> tuple of all type args from T as Base."""
    if len(typ.args) != 2:
        return UninhabitedType()

    args = _get_args(evaluator, typ.args[0], typ.args[1])

    if args is None:
        return UninhabitedType()
    return evaluator.tuple_type(list(args))


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
            # Expand the attribute type with the instance's type arguments
            return expand_type_by_instance(node.type, target)
        return UninhabitedType()

    return UninhabitedType()


@register_operator("Slice")
@lift_over_unions
def _eval_slice(evaluator: TypeLevelEvaluator, typ: TypeOperatorType) -> Type:
    """Evaluate Slice[S, Start, End] - slice a literal string or tuple type."""
    if len(typ.args) != 3:
        return UninhabitedType()

    target = evaluator.eval_proper(typ.args[0])

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

    # Handle literal string slicing
    s = extract_literal_string(target)
    if s is not None:
        result = s[start:end]
        return evaluator.literal_str(result)

    # Handle tuple type slicing
    if isinstance(target, TupleType):
        sliced_items = target.items[start:end]
        return evaluator.tuple_type(sliced_items)

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
    evaluator: TypeLevelEvaluator, target: TypedDictType, member_type_info: TypeInfo
) -> Type:
    """Evaluate Members/Attrs for a TypedDict type."""
    members: list[Type] = []

    for name, item_type in target.items.items():
        # Build qualifiers for TypedDict keys
        # Required is the default, so only add NotRequired when not required
        quals: list[str] = []
        if name not in target.required_keys:
            quals.append("NotRequired")
        if name in target.readonly_keys:
            quals.append("ReadOnly")

        quals_type = UnionType.make_union([evaluator.literal_str(q) for q in quals])

        member_type = Instance(
            member_type_info,
            [
                evaluator.literal_str(name),  # name
                item_type,  # typ
                quals_type,  # quals
                UninhabitedType(),  # init (not tracked for TypedDict)
                UninhabitedType(),  # definer (not tracked for TypedDict)
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


@register_operator("NewTypedDict")
def _eval_new_typeddict(evaluator: TypeLevelEvaluator, typ: TypeOperatorType) -> Type:
    """Evaluate NewTypedDict[*Members] -> create a new TypedDict from Member types.

    This is the inverse of Members[TypedDict].
    """
    # Get the Member TypeInfo to verify arguments
    member_info = evaluator.api.named_type_or_none("typing.Member")
    if member_info is None:
        return UninhabitedType()

    items: dict[str, Type] = {}
    required_keys: set[str] = set()
    readonly_keys: set[str] = set()

    for arg in evaluator.flatten_args(typ.args):
        arg = get_proper_type(arg)

        # Each argument should be a Member[name, typ, quals, init, definer]
        if not isinstance(arg, Instance) or arg.type != member_info.type:
            # Not a Member type - can't construct TypedDict
            return UninhabitedType()

        if len(arg.args) < 3:
            return UninhabitedType()

        # Extract name, type, and qualifiers from Member args
        name_type, item_type, quals, *_ = arg.args
        name = extract_literal_string(name_type)
        if name is None:
            return UninhabitedType()
        is_required = True  # Default is Required
        is_readonly = False

        # Check qualifiers - can be a single Literal or a Union of Literals
        quals_proper = get_proper_type(quals)
        qual_strings: list[str] = []

        if isinstance(quals_proper, LiteralType) and isinstance(quals_proper.value, str):
            qual_strings.append(quals_proper.value)
        elif isinstance(quals_proper, UnionType):
            for item in quals_proper.items:
                item_proper = get_proper_type(item)
                if isinstance(item_proper, LiteralType) and isinstance(item_proper.value, str):
                    qual_strings.append(item_proper.value)

        for qual in qual_strings:
            if qual == "NotRequired":
                is_required = False
            elif qual == "Required":
                is_required = True
            elif qual == "ReadOnly":
                is_readonly = True

        items[name] = item_type
        if is_required:
            required_keys.add(name)
        if is_readonly:
            readonly_keys.add(name)

    # Get the TypedDict fallback
    fallback = evaluator.api.named_type_or_none("typing._TypedDict")
    if fallback is None:
        # Fallback to Mapping[str, object] if _TypedDict not available
        fallback = evaluator.api.named_type("builtins.dict")

    return TypedDictType(
        items=items, required_keys=required_keys, readonly_keys=readonly_keys, fallback=fallback
    )


@register_operator("Length")
@lift_over_unions
def _eval_length(evaluator: TypeLevelEvaluator, typ: TypeOperatorType) -> Type:
    """Evaluate Length[T] -> Literal[int] for tuple length."""
    if len(typ.args) != 1:
        return UninhabitedType()

    target = evaluator.eval_proper(typ.args[0])

    if isinstance(target, TupleType):
        # Need to evaluate the elements before we inspect them
        items = [evaluator.eval_proper(st) for st in target.items]

        # If there is an Unpack, it must be of an unbounded tuple, or
        # it would have been substituted out.
        if any(isinstance(st, UnpackType) for st in items):
            return NoneType()
        return evaluator.literal_int(len(target.items))
    if isinstance(target, Instance) and target.type.has_base("builtins.tuple"):
        return NoneType()

    return UninhabitedType()


@register_operator("RaiseError")
def _eval_raise_error(evaluator: TypeLevelEvaluator, typ: TypeOperatorType) -> Type:
    """Evaluate RaiseError[S] -> emit a type error with message S.

    RaiseError is used to emit custom type errors during type-level computation.
    The argument must be a Literal[str] containing the error message.
    Returns Never after emitting the error.
    """

    args = evaluator.flatten_args(typ.args)
    if not args:
        msg = "RaiseError called without arguments!"
    else:
        msg = extract_literal_string(args[0]) or str(args[0])

    if args[1:]:
        msg += ": " + ", ".join(str(t) for t in args[1:])

    # TODO: We could also print a stack trace?
    ctx = evaluator.ctx or typ
    # Use serious=True to bypass in_checked_function() check which requires
    # self.options to be set on the SemanticAnalyzer
    evaluator.api.fail(msg, ctx, serious=True)

    return UninhabitedType()


def evaluate_comprehension(evaluator: TypeLevelEvaluator, typ: TypeForComprehension) -> Type:
    """Evaluate a TypeForComprehension.

    Evaluates *[Expr for var in Iter if Cond] to UnpackType(TupleType([...])).
    """

    # Get the iterable type and expand it to a TupleType
    iter_proper = evaluator.eval_proper(typ.iter_type)

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
        all_pass = True
        for cond in substituted_conditions:
            cond_result = extract_literal_bool(evaluator.evaluate(cond))
            if cond_result is False:
                all_pass = False
                break
            elif cond_result is None:
                # Undecidable condition - raise Stuck
                raise EvaluationStuck

        if all_pass:
            # Include this element in the result
            result_items.append(substituted_expr)

    return UnpackType(evaluator.tuple_type(result_items))


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


def evaluate_computed_type(typ: ComputedType, ctx: Context | None = None) -> Type:
    """Evaluate a ComputedType. Called from ComputedType.expand().

    Uses typelevel_ctx.api to access the semantic analyzer.

    The ctx argument indicates where an error message from RaiseError
    ought to be placed.  TODO: Make it a stack of contexts maybe?

    """
    if typelevel_ctx.api is None:
        raise AssertionError("No access to semantic analyzer!")

    old_evaluator = typelevel_ctx._evaluator
    if not typelevel_ctx._evaluator:
        typelevel_ctx._evaluator = TypeLevelEvaluator(typelevel_ctx.api, ctx)
    try:
        res = typelevel_ctx._evaluator.evaluate(typ)
    except EvaluationOverflow:
        # If this is not the top level of type evaluation, re-raise.
        if old_evaluator is not None:
            raise
        res = EXPANSION_OVERFLOW
    except EvaluationStuck:
        # TODO: Should we do the same top level thing as above?
        res = EXPANSION_ANY
    finally:
        typelevel_ctx._evaluator = old_evaluator

    # print("EVALED!!", typ, "====>", res)
    return res
