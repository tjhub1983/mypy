# Implementation Plan: Type-Level Computation for Mypy

This document outlines a plan for implementing the type-level computation proposal described in `../typemap/pre-pep.rst` and `../typemap/spec-draft.rst`.

## Overview

The proposal introduces TypeScript-inspired type-level introspection and construction facilities:

1. **Type Operators**: `GetArg`, `GetArgs`, `FromUnion`, `IsSub` (subtype check), `_Cond` (conditional)
2. **Conditional Types**: `_Cond[IsSub[T, Base], TrueType, FalseType]` (also supports ternary syntax `X if IsSub[T, Base] else Y`)
3. **Type-Level Iteration**: `*[... for t in Iter[...]]`
4. **Object Inspection**: `Members`, `Attrs`, `Member`, `NewProtocol`, `NewTypedDict`
5. **Callable Extension**: `Param` type with qualifiers for extended callable syntax
6. **String Operations**: `Slice`, `Concat`, `Uppercase`, `Lowercase`, etc.
7. **Annotated Operations**: `GetAnnotations`, `DropAnnotations`
8. **TypedDict `**kwargs` Inference**: `Unpack[K]` where K is a TypeVar bounded by TypedDict

---

## Phase 1: Foundation Types and Infrastructure

### 1.1 Core Design: Unified `TypeOperatorType` Class

Rather than creating a separate class for each type operator, we use a single unified
`TypeOperatorType` class modeled after `TypeAliasType` and `Instance`. This approach:

- Keeps mypy's type system minimal and extensible
- Allows new operators to be added in typeshed without modifying mypy's core
- Treats type operators as "unevaluated" types that expand to concrete types

### 1.2 Add `ComputedType` Base Class (`mypy/types.py`)

All type-level computation types share a common base class that defines the `expand()` interface:

```python
class ComputedType(Type):
    """
    Base class for types that represent unevaluated type-level computations.

    NOT a ProperType - must be expanded/evaluated before use in most type
    operations. Analogous to TypeAliasType in that it wraps a computation
    that produces a concrete type.

    Subclasses:
    - TypeOperatorType: e.g., GetArg[T, Base, 0], Members[T], _Cond[IsSub[T, Base], X, Y]
    - TypeForComprehension: e.g., *[Expr for x in Iter[T] if Cond]

    Note: Conditional types are represented as _Cond[...] TypeOperatorType, not a separate class.
    """

    __slots__ = ()

    def expand(self) -> Type:
        """
        Evaluate this computed type to produce a concrete type.
        Returns self if evaluation is not yet possible (e.g., contains unresolved type vars).

        Subclasses must implement this method.
        """
        raise NotImplementedError
```

### 1.3 Add `TypeOperatorType` (`mypy/types.py`)

```python
class TypeOperatorType(ComputedType):
    """
    Represents an unevaluated type operator application, e.g., GetArg[T, Base, 0].

    Stores a reference to the operator's TypeInfo and the type arguments.
    Type operators are generic classes in typeshed marked with @_type_operator.
    """

    __slots__ = ("type", "args")

    def __init__(
        self,
        type: TypeInfo,  # The TypeInfo for the operator (e.g., typing.GetArg)
        args: list[Type],  # The type arguments
        line: int = -1,
        column: int = -1,
    ) -> None:
        super().__init__(line, column)
        self.type = type
        self.args = args

    def accept(self, visitor: TypeVisitor[T]) -> T:
        return visitor.visit_type_operator_type(self)

    @property
    def fullname(self) -> str:
        return self.type.fullname

    def expand(self) -> Type:
        """Evaluate this type operator to produce a concrete type."""
        from mypy.typelevel import evaluate_type_operator
        return evaluate_type_operator(self)

    def serialize(self) -> JsonDict:
        return {
            ".class": "TypeOperatorType",
            "type_ref": self.type.fullname,
            "args": [a.serialize() for a in self.args],
        }

    @classmethod
    def deserialize(cls, data: JsonDict) -> TypeOperatorType:
        # Similar to TypeAliasType deserialization
        ...

    def copy_modified(self, *, args: list[Type] | None = None) -> TypeOperatorType:
        return TypeOperatorType(
            self.type,
            args if args is not None else self.args.copy(),
            self.line,
            self.column,
        )
```

### 1.4 Conditional Types and Comprehensions (`mypy/types.py`)

**Note:** Conditional types are now represented as `_Cond[condition, TrueType, FalseType]` using
`TypeOperatorType`, not a separate `ConditionalType` class. This simplifies the type system by
having one unified mechanism for all type-level computations. The ternary syntax
`X if IsSub[T, Base] else Y` is parsed and converted directly to `_Cond[IsSub[T, Base], X, Y]`.

```python
class TypeForComprehension(ComputedType):
    """
    Represents *[Expr for var in Iter[T] if Cond].

    Expands to a tuple of types.
    """

    __slots__ = ("element_expr", "iter_var", "iter_type", "conditions")

    def __init__(
        self,
        element_expr: Type,
        iter_var: str,
        iter_type: Type,  # The type being iterated (should be a tuple type)
        conditions: list[Type],  # Each should be IsSub[...] or boolean combo
        line: int = -1,
        column: int = -1,
    ) -> None:
        super().__init__(line, column)
        self.element_expr = element_expr
        self.iter_var = iter_var
        self.iter_type = iter_type
        self.conditions = conditions

    def accept(self, visitor: TypeVisitor[T]) -> T:
        return visitor.visit_type_for_comprehension(self)

    def expand(self) -> Type:
        """Evaluate the comprehension to produce a tuple type."""
        from mypy.typelevel import evaluate_comprehension
        return evaluate_comprehension(self)
```

### 1.5 Update Type Visitor (`mypy/type_visitor.py`)

Add visitor methods for the new types (note: no visitor for `ComputedType` base class - each subclass has its own):

```python
class TypeVisitor(Generic[T]):
    # ... existing methods ...

    def visit_type_operator_type(self, t: TypeOperatorType) -> T: ...
    def visit_type_for_comprehension(self, t: TypeForComprehension) -> T: ...
```

Note: There is no `visit_conditional_type` - conditional types are represented as `_Cond[...]`
TypeOperatorType and handled by `visit_type_operator_type`.

### 1.6 Declare Type Operators in Typeshed (`mypy/typeshed/stdlib/typing.pyi`)

All type operators are declared as generic classes with the `@_type_operator` decorator.
This decorator marks them for special handling by the type checker.

```python
# In typing.pyi

def _type_operator(cls: type[T]) -> type[T]: ...

# --- Data Types (used in type computations) ---

class Member(Generic[_Name, _Type, _Quals, _Init, _Definer]):
    """
    Represents a class member with name, type, qualifiers, initializer, and definer.
    - _Name: Literal[str] - the member name
    - _Type: the member's type
    - _Quals: Literal['ClassVar'] | Literal['Final'] | Never - qualifiers
    - _Init: the literal type of the initializer expression
    - _Definer: the class that defined this member
    """
    name: _Name
    typ: _Type
    quals: _Quals
    init: _Init
    definer: _Definer


class Param(Generic[_Name, _Type, _Quals]):
    """
    Represents a function parameter for extended callable syntax.
    - _Name: Literal[str] | None - the parameter name
    - _Type: the parameter's type
    - _Quals: Literal['positional', 'keyword', 'default', '*', '**'] - qualifiers
    """
    name: _Name
    typ: _Type
    quals: _Quals

# Convenience aliases for Param
type PosParam[N, T] = Param[N, T, Literal["positional"]]
type PosDefaultParam[N, T] = Param[N, T, Literal["positional", "default"]]
type DefaultParam[N, T] = Param[N, T, Literal["default"]]
type NamedParam[N, T] = Param[N, T, Literal["keyword"]]
type NamedDefaultParam[N, T] = Param[N, T, Literal["keyword", "default"]]
type ArgsParam[T] = Param[None, T, Literal["*"]]
type KwargsParam[T] = Param[None, T, Literal["**"]]

# --- Type Introspection Operators ---

@_type_operator
class GetArg(Generic[_T, _Base, _Idx]):
    """
    Get type argument at index _Idx from _T when viewed as _Base.
    Returns Never if _T does not inherit from _Base or index is out of bounds.
    """
    ...

@_type_operator
class GetArgs(Generic[_T, _Base]):
    """
    Get all type arguments from _T when viewed as _Base, as a tuple.
    Returns Never if _T does not inherit from _Base.
    """
    ...

@_type_operator
class GetAttr(Generic[_T, _Name]):
    """
    Get the type of attribute _Name from type _T.
    _Name must be a Literal[str].
    """
    ...

@_type_operator
class Members(Generic[_T]):
    """
    Get all members of type _T as a tuple of Member types.
    Includes methods, class variables, and instance attributes.
    """
    ...

@_type_operator
class Attrs(Generic[_T]):
    """
    Get annotated instance attributes of _T as a tuple of Member types.
    Excludes methods and ClassVar members.
    """
    ...

@_type_operator
class FromUnion(Generic[_T]):
    """
    Convert a union type to a tuple of its constituent types.
    If _T is not a union, returns a 1-tuple containing _T.
    """
    ...

# --- Member/Param Accessors (sugar for GetArg) ---

type GetName[T: Member | Param] = GetAttr[T, Literal["name"]]
type GetType[T: Member | Param] = GetAttr[T, Literal["typ"]]
type GetQuals[T: Member | Param] = GetAttr[T, Literal["quals"]]
type GetInit[T: Member] = GetAttr[T, Literal["init"]]
type GetDefiner[T: Member] = GetAttr[T, Literal["definer"]]

# --- Type Construction Operators ---

@_type_operator
class NewProtocol(Generic[Unpack[_Ts]]):
    """
    Construct a new structural (protocol) type from Member types.
    NewProtocol[Member[...], Member[...], ...] creates an anonymous protocol.
    """
    ...

@_type_operator
class NewTypedDict(Generic[Unpack[_Ts]]):
    """
    Construct a new TypedDict from Member types.
    NewTypedDict[Member[...], Member[...], ...] creates an anonymous TypedDict.
    """
    ...

# --- Boolean/Conditional Operators ---

@_type_operator
class IsSub(Generic[_T, _Base]):
    """
    Type-level subtype check. Evaluates to a type-level boolean.
    Used in conditional type expressions: `_Cond[IsSub[T, Base], TrueType, FalseType]`
    """
    ...

@_type_operator
class _Cond(Generic[_T, _TrueType, _FalseType]):
    """
    Type-level conditional expression.
    _Cond[IsSub[T, Base], TrueType, FalseType] evaluates to TrueType if T is a subtype of Base,
    otherwise FalseType.

    The ternary syntax `X if IsSub[T, Base] else Y` is converted to `_Cond[IsSub[T, Base], X, Y]`.
    """
    ...

@_type_operator
class Iter(Generic[_T]):
    """
    Marks a type for iteration in type comprehensions.
    `for x in Iter[T]` iterates over elements of tuple type T.
    """
    ...

# --- String Operations ---

@_type_operator
class Slice(Generic[_S, _Start, _End]):
    """
    Slice a literal string type.
    Slice[Literal["hello"], Literal[1], Literal[3]] = Literal["el"]
    """
    ...

@_type_operator
class Concat(Generic[_S1, _S2]):
    """
    Concatenate two literal string types.
    Concat[Literal["hello"], Literal["world"]] = Literal["helloworld"]
    """
    ...

@_type_operator
class Uppercase(Generic[_S]):
    """Convert literal string to uppercase."""
    ...

@_type_operator
class Lowercase(Generic[_S]):
    """Convert literal string to lowercase."""
    ...

@_type_operator
class Capitalize(Generic[_S]):
    """Capitalize first character of literal string."""
    ...

@_type_operator
class Uncapitalize(Generic[_S]):
    """Lowercase first character of literal string."""
    ...

# --- Annotated Operations ---

@_type_operator
class GetAnnotations(Generic[_T]):
    """
    Extract Annotated metadata from a type.
    GetAnnotations[Annotated[int, 'foo', 'bar']] = Literal['foo', 'bar']
    GetAnnotations[int] = Never
    """
    ...

@_type_operator
class DropAnnotations(Generic[_T]):
    """
    Strip Annotated wrapper from a type.
    DropAnnotations[Annotated[int, 'foo']] = int
    DropAnnotations[int] = int
    """
    ...

# --- Utility Operators ---

@_type_operator
class Length(Generic[_T]):
    """
    Get the length of a tuple type as a Literal[int].
    Returns Literal[None] for unbounded tuples.
    """
    ...
```

### 1.7 Detecting Type Operators (`mypy/nodes.py`)

Add a flag to TypeInfo to mark type operators:

```python
class TypeInfo(SymbolNode):
    # ... existing fields ...

    is_type_operator: bool = False  # True if decorated with @_type_operator
```

### 1.8 How Expansion Works

The key insight is that `ComputedType` (and its subclasses) is NOT a `ProperType`.
Like `TypeAliasType`, it must be expanded before most type operations can use it.
The expansion happens via:

1. **`get_proper_type()`** in `mypy/typeops.py` - already handles `TypeAliasType`, extend to handle `ComputedType`
2. **Explicit `.expand()` calls** when we need to evaluate

```python
# In mypy/typeops.py
def get_proper_type(typ: Type) -> ProperType:
    while True:
        if isinstance(typ, TypeAliasType):
            typ = typ._expand_once()
        elif isinstance(typ, ComputedType):
            # Handles TypeOperatorType (including _Cond), TypeForComprehension
            typ = typ.expand()
        else:
            break

    assert isinstance(typ, ProperType), type(typ)
    return typ
```

---

## Phase 2: Type Analysis (`mypy/typeanal.py`)

### 2.1 Detect and Construct TypeOperatorType

Instead of special-casing each operator, we detect classes marked with `@_type_operator`
and construct a generic `TypeOperatorType`:

```python
def analyze_unbound_type_nonoptional(
    self, t: UnboundType, report_invalid_types: bool
) -> Type:
    # ... existing logic to resolve the symbol ...

    node = self.lookup_qualified(t.name, t, ...)

    if isinstance(node, TypeInfo):
        # Check if this is a type operator
        if node.is_type_operator:
            return self.analyze_type_operator(t, node)

        # ... existing instance type handling ...

    # ... rest of existing logic ...


def analyze_type_operator(self, t: UnboundType, type_info: TypeInfo) -> Type:
    """
    Analyze a type operator application like GetArg[T, Base, 0].
    Returns a TypeOperatorType that will be expanded later.
    """
    # Analyze all type arguments
    args = [self.anal_type(arg) for arg in t.args]

    # Validate argument count against the operator's type parameters
    # (This is optional - could also defer to expansion time)
    expected = len(type_info.type_vars)
    if len(args) != expected:
        self.fail(
            f"Type operator {type_info.name} expects {expected} arguments, got {len(args)}",
            t
        )

    return TypeOperatorType(type_info, args, line=t.line, column=t.column)
```

### 2.2 Parse Conditional Type Syntax

Handle the `X if IsSub[T, Base] else Y` syntax in type contexts by extending the parser.
The ternary syntax is converted directly to `_Cond[condition, TrueType, FalseType]` TypeOperatorType.

#### 2.2.1 AST Representation

Python's parser already produces `IfExpr` (ternary) nodes. In type contexts, we need to
recognize these and convert them to `_Cond` type operator calls. The AST for `X if Cond else Y` is:

```python
IfExpr(
    cond=...,   # The condition expression
    body=...,   # The "true" branch (X)
    orelse=..., # The "false" branch (Y)
)
```

#### 2.2.2 Extend `expr_to_unanalyzed_type()` (`mypy/fastparse.py`)

The `expr_to_unanalyzed_type()` function converts AST expressions to unanalyzed types.
Extend it to handle `IfExpr`, converting to an UnboundType for `_Cond`:

```python
def expr_to_unanalyzed_type(
    expr: ast.expr,
    options: Options,
    ...,
) -> ProperType | UnboundType:
    # ... existing cases ...

    if isinstance(expr, IfExpr):
        # Convert ternary to _Cond[condition, true_type, false_type]
        condition = expr_to_unanalyzed_type(expr.cond, options, ...)
        true_type = expr_to_unanalyzed_type(expr.body, options, ...)
        false_type = expr_to_unanalyzed_type(expr.orelse, options, ...)
        return UnboundType(
            "builtins._Cond",
            [condition, true_type, false_type],
            line=expr.lineno,
            column=expr.col_offset,
        )

    # ... rest of existing logic ...
```

#### 2.2.3 Handle in Type Analysis (`mypy/typeanal.py`)

Since conditional types are now `_Cond[...]` TypeOperatorType, they are analyzed like
any other type operator via `analyze_type_operator()`. The condition validation happens
during evaluation in `mypy/typelevel.py`:

```python
def is_valid_type_condition(self, typ: Type) -> bool:
    """Check if typ is a valid type-level condition (IsSub or boolean combo)."""
    if isinstance(typ, TypeOperatorType):
        return typ.fullname == 'typing.IsSub'
    # Could also check for And/Or/Not combinations if we support those
    return False
```

### 2.3 Parse Type Comprehensions

Handle `*[Expr for var in Iter[T] if Cond]` within type argument lists:

```python
def analyze_starred_type_comprehension(self, expr: StarExpr) -> TypeForComprehension:
    """Analyze *[... for x in Iter[T] if ...]"""
    # This requires analyzing the list comprehension expression
    # and converting it to a TypeForComprehension
    pass
```

### 2.4 Extended Callable Parsing

Modify `analyze_callable_type()` to accept `Param` types in the argument list:

```python
def analyze_callable_type(self, t: UnboundType) -> Type:
    # ... existing logic ...

    # Check if args contain Param types (extended callable)
    if self.has_param_types(arg_types):
        return self.build_extended_callable(arg_types, ret_type)

    # ... existing logic ...
```

---

## Phase 3A: Core Conditional Types (`_Cond` and `IsSub`)

This phase implements the core conditional type evaluation - just `_Cond` and `IsSub`.
This is the foundation that enables conditional type expressions to work.

### 3A.1 Create Type Evaluator Core

```python
"""Type-level computation evaluation - Core conditional types."""

from __future__ import annotations

from typing import Callable

from mypy.types import (
    Type, ProperType, TypeOperatorType, TypeVarType,
)
from mypy.subtypes import is_subtype
from mypy.typeops import get_proper_type
from mypy.type_visitor import TypeQuery


# Registry mapping operator fullnames to their evaluation functions
_OPERATOR_EVALUATORS: dict[str, Callable[[TypeLevelEvaluator, TypeOperatorType], Type]] = {}


def register_operator(fullname: str):
    """Decorator to register an operator evaluator."""
    def decorator(func: Callable[[TypeLevelEvaluator, TypeOperatorType], Type]):
        _OPERATOR_EVALUATORS[fullname] = func
        return func
    return decorator


class TypeLevelEvaluator:
    """Evaluates type-level computations to concrete types."""

    def __init__(self, api: SemanticAnalyzerCoreInterface):
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
            # Unknown operator - return as-is (might be a data type like Member)
            return typ

        return evaluator(self, typ)

    def eval_condition(self, cond: Type) -> bool | None:
        """
        Evaluate a type-level condition (IsSub[T, Base]).
        Returns True/False if decidable, None if undecidable.
        """
        if isinstance(cond, TypeOperatorType) and cond.fullname == 'typing.IsSub':
            left = self.evaluate(cond.args[0])
            right = self.evaluate(cond.args[1])
            # Handle type variables - may be undecidable
            if self.contains_unresolved_typevar(left) or self.contains_unresolved_typevar(right):
                return None
            return is_subtype(left, right)

        # Could add support for boolean combinations (and, or, not) here
        return None

    def eval_conditional(self, typ: TypeOperatorType) -> Type:
        """Evaluate _Cond[condition, TrueType, FalseType]"""
        if len(typ.args) != 3:
            return typ
        condition, true_type, false_type = typ.args
        result = self.eval_condition(condition)
        if result is True:
            return self.evaluate(true_type)
        elif result is False:
            return self.evaluate(false_type)
        else:
            # Undecidable - keep as _Cond TypeOperatorType
            return typ

    def contains_unresolved_typevar(self, typ: Type) -> bool:
        """Check if type contains unresolved type variables."""

        class HasTypeVar(TypeQuery[bool]):
            def __init__(self):
                super().__init__(any)

            def visit_type_var(self, t: TypeVarType) -> bool:
                return True

        return typ.accept(HasTypeVar())


# --- Operator Implementations for Phase 3A ---

@register_operator('typing._Cond')
def eval_cond(evaluator: TypeLevelEvaluator, typ: TypeOperatorType) -> Type:
    """Evaluate _Cond[condition, TrueType, FalseType]"""
    return evaluator.eval_conditional(typ)


# Note: IsSub is not registered as an operator because it's not meant to be
# expanded directly - it's evaluated as a condition within _Cond.


# --- Public API ---

def evaluate_type_operator(typ: TypeOperatorType) -> Type:
    """Evaluate a TypeOperatorType. Called from TypeOperatorType.expand().

    Uses typelevel_ctx.api to access the semantic analyzer.
    """
    if typelevel_ctx.api is None:
        # No context available - can't evaluate yet
        return AnyType(TypeOfAny.special_form)

    evaluator = TypeLevelEvaluator(typelevel_ctx.api)
    return evaluator.eval_operator(typ)
```

---

## Phase 3B: Remaining Type Operators

This phase implements the remaining type operators after the core conditional types
are working. These include introspection, construction, and utility operators.

### 3B.1 Type Introspection Operators

Extend `TypeLevelEvaluator` with helper methods and implement the introspection operators:

```python
# --- Additional helper methods for TypeLevelEvaluator ---

class TypeLevelEvaluator:
    # ... methods from Phase 3A ...

    def get_type_args_for_base(self, instance: Instance, base: TypeInfo) -> list[Type] | None:
        """Get type args when viewing instance as base class."""
        for base_instance in instance.type.mro:
            if base_instance == base:
                return self.map_type_args_to_base(instance, base)
        return None

    def map_type_args_to_base(self, instance: Instance, base: TypeInfo) -> list[Type]:
        """Map instance's type args through inheritance chain to base."""
        from mypy.expandtype import expand_type_by_instance
        for b in instance.type.bases:
            b_proper = get_proper_type(b)
            if isinstance(b_proper, Instance) and b_proper.type == base:
                return list(expand_type_by_instance(b_proper, instance).args)
        return []

    def extract_literal_string(self, typ: Type) -> str | None:
        """Extract string value from LiteralType."""
        typ = get_proper_type(typ)
        if isinstance(typ, LiteralType) and isinstance(typ.value, str):
            return typ.value
        return None

    def extract_literal_int(self, typ: Type) -> int | None:
        """Extract int value from LiteralType."""
        typ = get_proper_type(typ)
        if isinstance(typ, LiteralType) and isinstance(typ.value, int):
            return typ.value
        return None

    def make_member_instance(
        self,
        name: str,
        member_type: Type,
        quals: Type,
        init: Type,
        definer: Type,
    ) -> Instance:
        """Create a Member[...] instance type."""
        member_info = self.api.lookup_qualified('typing.Member', ...).node
        return Instance(
            member_info,
            [
                LiteralType(name, self.api.named_type('builtins.str')),
                member_type,
                quals,
                init,
                definer,
            ],
        )

    def eval_comprehension(self, typ: TypeForComprehension) -> Type:
        """Evaluate *[Expr for x in Iter[T] if Cond]"""
        # First, evaluate the iter_type to get what we're iterating over
        iter_type = self.evaluate(typ.iter_type)

        # If it's an Iter[T] operator, extract T
        if isinstance(iter_type, TypeOperatorType) and iter_type.fullname == 'typing.Iter':
            iter_type = self.evaluate(iter_type.args[0])

        iter_type = get_proper_type(iter_type)

        if not isinstance(iter_type, TupleType):
            return typ  # Can't iterate over non-tuple

        results = []
        for item in iter_type.items:
            # Substitute iter_var with item in element_expr
            substituted = self.substitute_typevar(typ.element_expr, typ.iter_var, item)

            # Check conditions
            all_conditions_true = True
            for cond in typ.conditions:
                cond_subst = self.substitute_typevar(cond, typ.iter_var, item)
                result = self.eval_condition(cond_subst)
                if result is False:
                    all_conditions_true = False
                    break
                elif result is None:
                    # Undecidable - can't fully evaluate
                    return typ

            if all_conditions_true:
                results.append(self.evaluate(substituted))

        return TupleType(results, self.api.named_type('builtins.tuple'))

    # --- Helper methods ---

    def get_type_args_for_base(self, instance: Instance, base: TypeInfo) -> list[Type] | None:
        """Get type args when viewing instance as base class."""
        # Walk MRO to find base and map type arguments
        for base_instance in instance.type.mro:
            if base_instance == base:
                # Found it - now map arguments through inheritance
                return self.map_type_args_to_base(instance, base)
        return None

    def map_type_args_to_base(self, instance: Instance, base: TypeInfo) -> list[Type]:
        """Map instance's type args through inheritance chain to base."""
        from mypy.expandtype import expand_type_by_instance
        # Find the base in the MRO and get its type args
        for b in instance.type.bases:
            b_proper = get_proper_type(b)
            if isinstance(b_proper, Instance) and b_proper.type == base:
                return list(expand_type_by_instance(b_proper, instance).args)
        return []

    def contains_unresolved_typevar(self, typ: Type) -> bool:
        """Check if type contains unresolved type variables."""
        from mypy.types import TypeVarType
        from mypy.type_visitor import TypeQuery

        class HasTypeVar(TypeQuery[bool]):
            def __init__(self):
                super().__init__(any)

            def visit_type_var(self, t: TypeVarType) -> bool:
                return True

        return typ.accept(HasTypeVar())

    def substitute_typevar(self, typ: Type, var_name: str, replacement: Type) -> Type:
        """Substitute a type variable by name with a concrete type."""
        from mypy.type_visitor import TypeTranslator
        from mypy.types import TypeVarType

        class SubstituteVar(TypeTranslator):
            def visit_type_var(self, t: TypeVarType) -> Type:
                if t.name == var_name:
                    return replacement
                return t

        return typ.accept(SubstituteVar())

    def extract_literal_string(self, typ: Type) -> str | None:
        """Extract string value from LiteralType."""
        typ = get_proper_type(typ)
        if isinstance(typ, LiteralType) and isinstance(typ.value, str):
            return typ.value
        return None

    def extract_literal_int(self, typ: Type) -> int | None:
        """Extract int value from LiteralType."""
        typ = get_proper_type(typ)
        if isinstance(typ, LiteralType) and isinstance(typ.value, int):
            return typ.value
        return None

    def make_member_instance(
        self,
        name: str,
        member_type: Type,
        quals: Type,
        init: Type,
        definer: Type,
    ) -> Instance:
        """Create a Member[...] instance type (Member is a regular generic class)."""
        member_info = self.api.lookup_qualified('typing.Member', ...).node
        return Instance(
            member_info,
            [
                LiteralType(name, self.api.named_type('builtins.str')),
                member_type,
                quals,
                init,
                definer,
            ],
        )

    def create_protocol_from_members(self, members: list[Instance]) -> Type:
        """Create a new Protocol TypeInfo from Member type operators."""
        # Extract member info and create synthetic TypeInfo
        # This is complex - see Phase 4 for details
        pass


# --- Operator Implementations ---

@register_operator('builtins._Cond')
def eval_cond(evaluator: TypeLevelEvaluator, typ: TypeOperatorType) -> Type:
    """Evaluate _Cond[condition, TrueType, FalseType]"""
    return evaluator.eval_conditional(typ)


@register_operator('typing.GetArg')
def eval_get_arg(evaluator: TypeLevelEvaluator, typ: TypeOperatorType) -> Type:
    """Evaluate GetArg[T, Base, Idx]"""
    if len(typ.args) != 3:
        return typ

    target = evaluator.evaluate(typ.args[0])
    base = evaluator.evaluate(typ.args[1])
    idx = evaluator.evaluate(typ.args[2])

    target = get_proper_type(target)
    base = get_proper_type(base)

    # Extract index as int
    index = evaluator.extract_literal_int(idx)
    if index is None:
        return typ  # Can't evaluate without literal index

    if isinstance(target, Instance) and isinstance(base, Instance):
        # This works for both regular classes and Member/Param (which are now Instances)
        args = evaluator.get_type_args_for_base(target, base.type)
        if args is not None and 0 <= index < len(args):
            return args[index]
        return UninhabitedType()  # Never

    return typ


@register_operator('typing.GetArgs')
def eval_get_args(evaluator: TypeLevelEvaluator, typ: TypeOperatorType) -> Type:
    """Evaluate GetArgs[T, Base] -> tuple of args"""
    if len(typ.args) != 2:
        return typ

    target = evaluator.evaluate(typ.args[0])
    base = evaluator.evaluate(typ.args[1])

    target = get_proper_type(target)
    base = get_proper_type(base)

    if isinstance(target, Instance) and isinstance(base, Instance):
        args = evaluator.get_type_args_for_base(target, base.type)
        if args is not None:
            return TupleType(list(args), evaluator.api.named_type('builtins.tuple'))
        return UninhabitedType()

    return typ


@register_operator('typing.Members')
def eval_members(evaluator: TypeLevelEvaluator, typ: TypeOperatorType) -> Type:
    """Evaluate Members[T] -> tuple of Member instance types"""
    if len(typ.args) != 1:
        return typ

    target = evaluator.evaluate(typ.args[0])
    target = get_proper_type(target)

    if isinstance(target, Instance):
        members = []
        for name, node in target.type.names.items():
            if node.type is not None:
                member = evaluator.make_member_instance(
                    name=name,
                    member_type=node.type,
                    quals=extract_member_quals(node),
                    init=extract_member_init(node),
                    definer=Instance(target.type, []),
                )
                members.append(member)
        return TupleType(members, evaluator.api.named_type('builtins.tuple'))

    return typ


@register_operator('typing.Attrs')
def eval_attrs(evaluator: TypeLevelEvaluator, typ: TypeOperatorType) -> Type:
    """Evaluate Attrs[T] -> tuple of Member instance types (annotated attrs only)"""
    if len(typ.args) != 1:
        return typ

    target = evaluator.evaluate(typ.args[0])
    target = get_proper_type(target)

    if isinstance(target, Instance):
        members = []
        for name, node in target.type.names.items():
            # Filter to annotated instance attributes only
            if (node.type is not None and
                not node.is_classvar and
                not isinstance(node.type, CallableType)):
                member = evaluator.make_member_instance(
                    name=name,
                    member_type=node.type,
                    quals=extract_member_quals(node),
                    init=extract_member_init(node),
                    definer=Instance(target.type, []),
                )
                members.append(member)
        return TupleType(members, evaluator.api.named_type('builtins.tuple'))

    return typ


@register_operator('typing.FromUnion')
def eval_from_union(evaluator: TypeLevelEvaluator, typ: TypeOperatorType) -> Type:
    """Evaluate FromUnion[T] -> tuple of union elements"""
    if len(typ.args) != 1:
        return typ

    target = evaluator.evaluate(typ.args[0])
    target = get_proper_type(target)

    if isinstance(target, UnionType):
        return TupleType(list(target.items), evaluator.api.named_type('builtins.tuple'))
    else:
        # Non-union becomes 1-tuple
        return TupleType([target], evaluator.api.named_type('builtins.tuple'))


@register_operator('typing.GetAttr')
def eval_get_attr(evaluator: TypeLevelEvaluator, typ: TypeOperatorType) -> Type:
    """Evaluate GetAttr[T, Name]"""
    if len(typ.args) != 2:
        return typ

    target = evaluator.evaluate(typ.args[0])
    name_type = evaluator.evaluate(typ.args[1])

    target = get_proper_type(target)
    name = evaluator.extract_literal_string(name_type)

    if name is None:
        return typ

    if isinstance(target, Instance):
        node = target.type.names.get(name)
        if node is not None and node.type is not None:
            return node.type
        return UninhabitedType()

    return typ


# --- Member/Param Accessors ---
# NOTE: GetName, GetType, GetQuals, GetInit, GetDefiner are now type aliases
# defined in typeshed using GetAttr, not type operators:
#
#   type GetName[T: Member | Param] = GetAttr[T, Literal["name"]]
#   type GetType[T: Member | Param] = GetAttr[T, Literal["typ"]]
#   type GetQuals[T: Member | Param] = GetAttr[T, Literal["quals"]]
#   type GetInit[T: Member] = GetAttr[T, Literal["init"]]
#   type GetDefiner[T: Member] = GetAttr[T, Literal["definer"]]
#
# Since Member and Param are regular generic classes with attributes,
# GetAttr handles these automatically - no special operator needed.


# --- String Operations ---

@register_operator('typing.Slice')
def eval_slice(evaluator: TypeLevelEvaluator, typ: TypeOperatorType) -> Type:
    """Evaluate Slice[S, Start, End]"""
    if len(typ.args) != 3:
        return typ

    s = evaluator.extract_literal_string(evaluator.evaluate(typ.args[0]))
    start = evaluator.extract_literal_int(evaluator.evaluate(typ.args[1]))
    end = evaluator.extract_literal_int(evaluator.evaluate(typ.args[2]))

    # Handle None for start/end
    start_arg = get_proper_type(evaluator.evaluate(typ.args[1]))
    end_arg = get_proper_type(evaluator.evaluate(typ.args[2]))
    if isinstance(start_arg, NoneType):
        start = None
    if isinstance(end_arg, NoneType):
        end = None

    if s is not None:
        result = s[start:end]
        return LiteralType(result, evaluator.api.named_type('builtins.str'))

    return typ


@register_operator('typing.Concat')
def eval_concat(evaluator: TypeLevelEvaluator, typ: TypeOperatorType) -> Type:
    """Evaluate Concat[S1, S2]"""
    if len(typ.args) != 2:
        return typ

    left = evaluator.extract_literal_string(evaluator.evaluate(typ.args[0]))
    right = evaluator.extract_literal_string(evaluator.evaluate(typ.args[1]))

    if left is not None and right is not None:
        return LiteralType(left + right, evaluator.api.named_type('builtins.str'))

    return typ


@register_operator('typing.Uppercase')
def eval_uppercase(evaluator: TypeLevelEvaluator, typ: TypeOperatorType) -> Type:
    if len(typ.args) != 1:
        return typ
    s = evaluator.extract_literal_string(evaluator.evaluate(typ.args[0]))
    if s is not None:
        return LiteralType(s.upper(), evaluator.api.named_type('builtins.str'))
    return typ


@register_operator('typing.Lowercase')
def eval_lowercase(evaluator: TypeLevelEvaluator, typ: TypeOperatorType) -> Type:
    if len(typ.args) != 1:
        return typ
    s = evaluator.extract_literal_string(evaluator.evaluate(typ.args[0]))
    if s is not None:
        return LiteralType(s.lower(), evaluator.api.named_type('builtins.str'))
    return typ


@register_operator('typing.Capitalize')
def eval_capitalize(evaluator: TypeLevelEvaluator, typ: TypeOperatorType) -> Type:
    if len(typ.args) != 1:
        return typ
    s = evaluator.extract_literal_string(evaluator.evaluate(typ.args[0]))
    if s is not None:
        return LiteralType(s.capitalize(), evaluator.api.named_type('builtins.str'))
    return typ


@register_operator('typing.Uncapitalize')
def eval_uncapitalize(evaluator: TypeLevelEvaluator, typ: TypeOperatorType) -> Type:
    if len(typ.args) != 1:
        return typ
    s = evaluator.extract_literal_string(evaluator.evaluate(typ.args[0]))
    if s is not None:
        result = s[0].lower() + s[1:] if s else s
        return LiteralType(result, evaluator.api.named_type('builtins.str'))
    return typ


# --- Type Construction ---

@register_operator('typing.NewProtocol')
def eval_new_protocol(evaluator: TypeLevelEvaluator, typ: TypeOperatorType) -> Type:
    """Evaluate NewProtocol[*Members] -> create a new structural type"""
    evaluated_members = [evaluator.evaluate(m) for m in typ.args]

    # All members must be Member instances (Member is a regular generic class)
    member_type_info = evaluator.api.lookup_qualified('typing.Member', ...).node
    for m in evaluated_members:
        m = get_proper_type(m)
        if not isinstance(m, Instance) or m.type != member_type_info:
            return typ  # Can't evaluate yet

    return evaluator.create_protocol_from_members(evaluated_members)


@register_operator('typing.NewTypedDict')
def eval_new_typed_dict(evaluator: TypeLevelEvaluator, typ: TypeOperatorType) -> Type:
    """Evaluate NewTypedDict[*Members] -> create a new TypedDict"""
    evaluated_members = [evaluator.evaluate(m) for m in typ.args]

    member_type_info = evaluator.api.lookup_qualified('typing.Member', ...).node
    items = {}
    required_keys = set()

    for m in evaluated_members:
        m = get_proper_type(m)
        if not isinstance(m, Instance) or m.type != member_type_info:
            return typ  # Can't evaluate yet

        # Member[name, typ, quals, init, definer] - access via type args
        name = evaluator.extract_literal_string(m.args[0])
        if name is None:
            return typ

        items[name] = m.args[1]  # The type
        # Check quals (args[2]) for Required/NotRequired
        quals = get_proper_type(m.args[2]) if len(m.args) > 2 else UninhabitedType()
        if not has_not_required_qual(quals):
            required_keys.add(name)

    return TypedDictType(
        items=items,
        required_keys=required_keys,
        readonly_keys=frozenset(),
        fallback=evaluator.api.named_type('typing.TypedDict'),
    )


@register_operator('typing.Length')
def eval_length(evaluator: TypeLevelEvaluator, typ: TypeOperatorType) -> Type:
    """Evaluate Length[T] -> Literal[int] for tuple length"""
    if len(typ.args) != 1:
        return typ

    target = evaluator.evaluate(typ.args[0])
    target = get_proper_type(target)

    if isinstance(target, TupleType):
        if target.partial_fallback:
            # Unbounded tuple
            return NoneType()
        return LiteralType(len(target.items), evaluator.api.named_type('builtins.int'))

    return typ


# --- Helper functions ---

def extract_member_quals(node) -> Type:
    """Extract qualifiers (ClassVar, Final) from a symbol table node."""
    # Implementation depends on how qualifiers are stored
    return UninhabitedType()  # Never = no qualifiers


def extract_member_init(node) -> Type:
    """Extract the literal type of an initializer from a symbol table node."""
    # Implementation depends on how initializers are tracked
    return UninhabitedType()  # Never = no initializer


def has_not_required_qual(quals: Type) -> bool:
    """Check if qualifiers include NotRequired."""
    # Implementation depends on qualifier representation
    return False


# --- Public API ---

def evaluate_type_operator(typ: TypeOperatorType) -> Type:
    """Evaluate a TypeOperatorType. Called from TypeOperatorType.expand().

    This handles all type operators including _Cond for conditional types.
    """
    # Need to get the API somehow - this is a design question
    # Option 1: Pass API through a context variable
    # Option 2: Store API reference on TypeOperatorType
    # Option 3: Create evaluator lazily
    evaluator = TypeLevelEvaluator(...)
    return evaluator.eval_operator(typ)


def evaluate_comprehension(typ: TypeForComprehension) -> Type:
    """Evaluate a TypeForComprehension. Called from TypeForComprehension.expand()."""
    evaluator = TypeLevelEvaluator(...)
    return evaluator.eval_comprehension(typ)
```

---

## Phase 4: Integration Points

### 4.1 Integrate with Type Alias Expansion (`mypy/types.py`)

Modify `TypeAliasType._expand_once()` to evaluate type-level computations:

```python
def _expand_once(self) -> Type:
    # ... existing expansion logic ...

    # After substitution, evaluate type-level computations
    if self.alias is not None:
        result = expand_type(self.alias.target, type_env)
        evaluator = TypeLevelEvaluator(...)
        result = evaluator.evaluate(result)
        return result
```

### 4.2 Integrate with `expand_type()` (`mypy/expandtype.py`)

Extend `ExpandTypeVisitor` to handle new types:

```python
class ExpandTypeVisitor(TypeTransformVisitor):
    # ... existing methods ...

    def visit_type_operator_type(self, t: TypeOperatorType) -> Type:
        # Expand type args, including _Cond which has condition, true_type, false_type as args
        return t.copy_modified(args=[arg.accept(self) for arg in t.args])

    def visit_type_for_comprehension(self, t: TypeForComprehension) -> Type:
        # Don't substitute the iteration variable
        return TypeForComprehension(
            t.element_expr.accept(self),
            t.iter_var,
            t.iter_type.accept(self),
            [c.accept(self) for c in t.conditions],
        )

    # ... more visit methods for other new types ...
```

Note: Conditional types are now `_Cond[...]` TypeOperatorType, so they are handled by
`visit_type_operator_type` along with all other type operators.

### 4.3 Subtype Checking (`mypy/subtypes.py`)

Add subtype rules for new types:

```python
class SubtypeVisitor(TypeVisitor[bool]):
    # ... existing methods ...

    def visit_type_operator_type(self, left: TypeOperatorType) -> bool:
        # For _Cond[condition, TrueType, FalseType]: subtype if both branches are subtypes
        # OR if we can evaluate the condition
        if left.fullname == 'builtins._Cond' and len(left.args) == 3:
            condition, true_type, false_type = left.args
            evaluator = TypeLevelEvaluator(...)
            result = evaluator.eval_condition(condition)

            if result is True:
                return is_subtype(true_type, self.right)
            elif result is False:
                return is_subtype(false_type, self.right)
            else:
                # Must be subtype in both cases
                return (is_subtype(true_type, self.right) and
                        is_subtype(false_type, self.right))

        # For other type operators, expand first then check subtype
        expanded = left.expand()
        if expanded is not left:
            return is_subtype(expanded, self.right)
        return False  # Unevaluatable type operator
```

### 4.4 Type Inference with `**kwargs` TypeVar

Handle `Unpack[K]` where K is bounded by TypedDict:

In `mypy/checkexpr.py`, extend `check_call_expr_with_callee_type()`:

```python
def infer_typeddict_from_kwargs(
    self,
    callee: CallableType,
    kwargs: dict[str, Expression],
) -> dict[TypeVarId, Type]:
    """Infer TypedDict type from **kwargs when unpacking a TypeVar."""
    # Find if callee has **kwargs: Unpack[K] where K is TypeVar
    # Build TypedDict from provided kwargs and their inferred types
    pass
```

---

## Phase 5: Extended Callable Support

### 5.1 `Param` Type for Callable Arguments

Support `Callable[[Param[N, T, Q], ...], R]` syntax:

```python
# In typeanal.py
def build_extended_callable(
    self,
    params: list[ParamType],
    ret_type: Type,
) -> CallableType:
    """Build CallableType from Param types."""
    arg_types = []
    arg_kinds = []
    arg_names = []

    for param in params:
        arg_types.append(param.param_type)
        arg_names.append(self.extract_param_name(param))
        arg_kinds.append(self.extract_param_kind(param))

    return CallableType(
        arg_types=arg_types,
        arg_kinds=arg_kinds,
        arg_names=arg_names,
        ret_type=ret_type,
        fallback=self.api.named_type('builtins.function'),
    )
```

### 5.2 Expose Callables as Extended Format

When introspecting `Callable` via `Members` or similar, expose params as `Param` types:

```python
def callable_to_param_types(self, callable: CallableType) -> list[ParamType]:
    """Convert CallableType to list of ParamType."""
    params = []
    for i, (typ, kind, name) in enumerate(zip(
        callable.arg_types, callable.arg_kinds, callable.arg_names
    )):
        quals = self.kind_to_param_quals(kind)
        name_type = LiteralType(name, ...) if name else NoneType()
        params.append(ParamType(name_type, typ, quals))
    return params
```

---

## Phase 6: Annotated Operations

### 6.1 `GetAnnotations[T]`

Extract `Annotated` metadata:

```python
def eval_get_annotations(self, typ: GetAnnotationsType) -> Type:
    target = self.evaluate(typ.target)
    # Note: This requires changes to how we store Annotated types
    # Currently mypy strips annotations - we need to preserve them

    if hasattr(target, '_annotations'):
        # Return as union of Literal types
        return UnionType.make_union([
            LiteralType(a, ...) for a in target._annotations
        ])
    return UninhabitedType()  # Never
```

### 6.2 Preserve Annotations in Type Representation

Modify `analyze_annotated_type()` in `typeanal.py` to preserve annotation metadata:

```python
class AnnotatedType(ProperType):
    """Represents Annotated[T, ann1, ann2, ...]"""
    inner_type: Type
    annotations: tuple[Any, ...]
```

---

## Phase 7: InitField Support

### 7.1 `InitField` Type for Field Descriptors

Support literal type inference for field initializers:

```python
# In semanal.py, when analyzing class body assignments
def analyze_class_attribute_with_initfield(
    self,
    name: str,
    typ: Type,
    init_expr: Expression,
) -> None:
    """Handle `attr: T = InitField(...)` patterns."""
    if self.is_initfield_call(init_expr):
        # Infer literal types for all kwargs
        kwargs_types = self.infer_literal_kwargs(init_expr)
        # Store as Member init type
        init_type = self.create_initfield_literal_type(kwargs_types)
        # ... store in symbol table
```

---

## Phase 8: Testing Strategy

### 8.1 Unit Tests

Create comprehensive tests in `mypy/test/`:

1. **`test_typelevel_basic.py`** - Basic type operators
2. **`test_typelevel_conditional.py`** - Conditional types
3. **`test_typelevel_comprehension.py`** - Type comprehensions
4. **`test_typelevel_protocol.py`** - NewProtocol creation
5. **`test_typelevel_typeddict.py`** - NewTypedDict creation
6. **`test_typelevel_callable.py`** - Extended callable / Param
7. **`test_typelevel_string.py`** - String operations
8. **`test_typelevel_examples.py`** - Full examples from PEP

### 8.2 Test Data Files

Create `.test` files for each feature area:

```
test-data/unit/check-typelevel-getarg.test
test-data/unit/check-typelevel-conditional.test
test-data/unit/check-typelevel-members.test
test-data/unit/check-typelevel-newprotocol.test
...
```

### 8.3 Integration Tests

Port examples from the PEP:
- Prisma-style ORM query builder
- FastAPI CRUD model derivation
- Dataclass-style `__init__` generation

---

## Phase 9: Incremental Implementation Order

### Milestone 1: Foundation (Weeks 1-2) ✓ COMPLETED
1. Add `ComputedType` base class, `TypeOperatorType`, `TypeForComprehension`
2. Add `is_type_operator` flag to `TypeInfo`
3. Declare type operators in typeshed with `@_type_operator`
4. Update type visitors for new types

### Milestone 2: Type Analysis (Weeks 3-4) ✓ COMPLETED
1. Detect `@_type_operator` decorated classes in semanal.py
2. Construct `TypeOperatorType` in typeanal.py when encountering type operators
3. Parse ternary syntax `X if Cond else Y` to `_Cond[Cond, X, Y]`
4. Add context variable for API access (`typelevel_ctx`)
5. Tests for type operator detection and ternary parsing

### Milestone 3A: Core Conditional Evaluation (Week 5)
1. Implement `TypeLevelEvaluator` core with `eval_condition` and `eval_conditional`
2. Register `typing._Cond` operator
3. Implement `IsSub` condition evaluation using `is_subtype()`
4. Wire up `typelevel_ctx` in type analysis
5. Tests for conditional type evaluation

### Milestone 3B: Introspection Operators (Weeks 6-7)
1. Add `GetArg`, `GetArgs`, `FromUnion` operators
2. Add `GetAttr` operator
3. Add `Members`, `Attrs` operators
4. Tests for introspection operators

### Milestone 4: Type Comprehensions (Weeks 8-9)
1. Add `TypeForComprehension` evaluation
2. Parser support for comprehension syntax `*[... for x in Iter[T]]`
3. Add `Iter` operator support
4. Tests for comprehensions

### Milestone 5: Type Construction (Weeks 10-12)
1. Add `NewProtocol` - synthetic protocol creation
2. Add `NewTypedDict` - synthetic TypedDict creation
3. Integration with type checking
4. Tests for type construction

### Milestone 6: Extended Callables (Weeks 13-14)
1. Full `Param` type support
2. Callable introspection
3. Extended callable construction
4. Tests for callables

### Milestone 7: String Operations (Week 15)
1. Add `Slice`, `Concat` operators
2. Add `Uppercase`, `Lowercase`, `Capitalize`, `Uncapitalize`
3. Tests for string ops

### Milestone 8: Annotated & InitField (Weeks 16-17)
1. Preserve Annotated metadata
2. `GetAnnotations`/`DropAnnotations`
3. InitField support
4. Tests

### Milestone 9: TypedDict kwargs inference (Week 18)
1. `Unpack[K]` for TypeVar K
2. Inference from kwargs
3. Tests

### Milestone 10: Integration & Polish (Weeks 19-20)
1. Full PEP examples working
2. Error messages
3. Documentation
4. Performance optimization

---

## Key Design Decisions

### 1. Unified TypeOperatorType (Not Per-Operator Classes)
**Decision**: Use a single `TypeOperatorType` class that references a TypeInfo (the operator) and contains args, rather than creating separate classes for each operator (GetArgType, MembersType, etc.). This keeps mypy's core minimal and allows new operators to be added in typeshed without modifying mypy.

### 2. Type Operators Declared in Typeshed
**Decision**: Type operators (`GetArg`, `Members`, `GetAttr`, etc.) are declared as generic classes in `mypy/typeshed/stdlib/typing.pyi` with the `@_type_operator` decorator. `Member` and `Param` are regular generic classes (not operators) with actual attributes - they're just data containers used in type computations.

### 2b. Member/Param Accessors as Type Aliases
**Decision**: `GetName`, `GetType`, `GetQuals`, `GetInit`, `GetDefiner` are type aliases using `GetAttr`, not separate type operators. Since `Member` and `Param` are regular classes with attributes, `GetAttr[Member[...], Literal["name"]]` works directly.

### 3. ComputedType Hierarchy (NOT ProperType)
**Decision**: All computed types (`TypeOperatorType`, `TypeForComprehension`) inherit from a common `ComputedType` base class. Like `TypeAliasType`, `ComputedType` is NOT a `ProperType` and must be expanded before use in most type operations. This is handled by a single `isinstance(typ, ComputedType)` check in `get_proper_type()`. Note: Conditional types are represented as `_Cond[...]` TypeOperatorType, not a separate class.

### 4. Lazy Evaluation with Caching
**Decision**: Type-level computations are evaluated when needed (e.g., during subtype checking) rather than immediately during parsing. Results should be cached.

### 5. Handling Undecidable Conditions
**Decision**: When a condition cannot be evaluated (e.g., involves unbound type variables), preserve the conditional type. It will be evaluated later when more type information is available.

### 6. Synthetic Type Identity
**Decision**: Types created via `NewProtocol` are structural (protocols), so identity is based on structure, not name. Each creation point may produce a "different" type that is structurally equivalent.

### 7. Error Handling
**Decision**: Invalid type-level operations (e.g., `GetArg` on non-generic type) return `Never` rather than raising errors, consistent with the spec.

### 8. Runtime Evaluation
**Decision**: This implementation focuses on static type checking. Runtime evaluation is a separate library concern (as noted in the spec).

### 9. Registry-Based Operator Dispatch
**Decision**: The evaluator uses a registry mapping operator fullnames to evaluation functions (via `@register_operator` decorator). This allows adding new operators without modifying the core evaluator logic.

---

## Files to Create/Modify

### New Files
- `mypy/typelevel.py` - Type-level computation evaluator with operator registry
- `mypy/test/test_typelevel_*.py` - Test files
- `test-data/unit/check-typelevel-*.test` - Test data

### Modified Files
- `mypy/types.py` - Add `ComputedType` base class, `TypeOperatorType`, `TypeForComprehension`
- `mypy/type_visitor.py` - Add `visit_type_operator_type`, `visit_type_for_comprehension`
- `mypy/fastparse.py` - Extend `expr_to_unanalyzed_type()` to handle `IfExpr` → `_Cond[...]` UnboundType
- `mypy/typeanal.py` - Detect `@_type_operator` classes, construct `TypeOperatorType`
- `mypy/typeops.py` - Extend `get_proper_type()` to expand type operators
- `mypy/expandtype.py` - Handle type variable substitution in type operators
- `mypy/subtypes.py` - Subtype rules for unevaluated type operators
- `mypy/checkexpr.py` - kwargs TypedDict inference
- `mypy/semanal.py` - Detect `@_type_operator` decorator, InitField handling
- `mypy/nodes.py` - Add `is_type_operator` flag to `TypeInfo`
- `mypy/typeshed/stdlib/typing.pyi` - Declare type operators with `@_type_operator` (including `_Cond` for conditionals), plus `Member`/`Param` as regular generic classes and accessor aliases

---

## Open Questions for Discussion

1. **Protocol vs TypedDict creation**: Should `NewProtocol` create true protocols (with `is_protocol=True`) or just structural types?

2. **Type alias recursion**: How to handle recursive type aliases that use type-level computation?

3. **Error recovery**: What should happen when type-level computation fails? Currently spec says return `Never`.

4. **Caching strategy**: How aggressively to cache evaluated type-level computations?

5. **API access in expand()**: ~~How does `TypeOperatorType.expand()` get access to the semantic analyzer API? Options: context variable, stored reference, or lazy creation.~~
   **RESOLVED**: Use a context variable (`typelevel_ctx` in `mypy/typelevel.py`). The `TypeLevelContext` class holds a reference to the `SemanticAnalyzerCoreInterface` API, set via a context manager (`typelevel_ctx.set_api(api)`) during type analysis. The evaluation functions access it via `typelevel_ctx.api`.

6. **Type variable handling in operators**: When should type variables in operator arguments block evaluation vs. be substituted first?
