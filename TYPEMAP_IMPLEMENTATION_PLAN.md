# Implementation Plan: Type-Level Computation for Mypy

This document outlines a plan for implementing the type-level computation proposal described in `../typemap/pre-pep.rst` and `../typemap/spec-draft.rst`.

## Overview

The proposal introduces TypeScript-inspired type-level introspection and construction facilities:

1. **Type Operators**: `GetArg`, `GetArgs`, `FromUnion`, `Sub` (subtype check)
2. **Conditional Types**: `X if Sub[T, Base] else Y`
3. **Type-Level Iteration**: `*[... for t in Iter[...]]`
4. **Object Inspection**: `Members`, `Attrs`, `Member`, `NewProtocol`, `NewTypedDict`
5. **Callable Extension**: `Param` type with qualifiers for extended callable syntax
6. **String Operations**: `Slice`, `Concat`, `Uppercase`, `Lowercase`, etc.
7. **Annotated Operations**: `GetAnnotations`, `DropAnnotations`
8. **TypedDict `**kwargs` Inference**: `Unpack[K]` where K is a TypeVar bounded by TypedDict

---

## Phase 1: Foundation Types and Infrastructure

### 1.1 Add Core Type Classes (`mypy/types.py`)

#### New Type Classes

```python
class MemberType(ProperType):
    """Represents Member[Name, Type, Quals, Init, Definer]"""
    name: Type  # Literal string type
    member_type: Type
    quals: Type  # Literal union of qualifiers
    init: Type  # Literal type of initializer
    definer: Type  # The class that defined this member

class ParamType(ProperType):
    """Represents Param[Name, Type, Quals] for extended callables"""
    name: Type  # Literal string or None
    param_type: Type
    quals: Type  # Literal union of param qualifiers

class ConditionalType(ProperType):
    """Represents `TrueType if Condition else FalseType`"""
    condition: TypeCondition
    true_type: Type
    false_type: Type

class TypeCondition:
    """Base class for type-level boolean conditions"""
    pass

class SubtypeCondition(TypeCondition):
    """Represents Sub[T, Base] - a subtype check"""
    left: Type
    right: Type

class NotCondition(TypeCondition):
    """Represents `not <condition>`"""
    inner: TypeCondition

class AndCondition(TypeCondition):
    """Represents `cond1 and cond2`"""
    left: TypeCondition
    right: TypeCondition

class OrCondition(TypeCondition):
    """Represents `cond1 or cond2`"""
    left: TypeCondition
    right: TypeCondition

class TypeForComprehension(ProperType):
    """Represents *[Expr for var in Iter[T] if Cond]"""
    element_expr: Type
    iter_var: str
    iter_type: Type  # The type being iterated
    conditions: list[TypeCondition]

class TypeOperatorType(ProperType):
    """Base class for type operators like GetArg, GetAttr, etc."""
    pass

class GetArgType(TypeOperatorType):
    """Represents GetArg[T, Base, Idx]"""
    target: Type
    base: Type
    index: Type  # Literal int

class GetArgsType(TypeOperatorType):
    """Represents GetArgs[T, Base]"""
    target: Type
    base: Type

class GetAttrType(TypeOperatorType):
    """Represents GetAttr[T, AttrName]"""
    target: Type
    attr_name: Type  # Literal string

class MembersType(TypeOperatorType):
    """Represents Members[T]"""
    target: Type

class AttrsType(TypeOperatorType):
    """Represents Attrs[T] - annotated attributes only"""
    target: Type

class FromUnionType(TypeOperatorType):
    """Represents FromUnion[T]"""
    target: Type

class NewProtocolType(TypeOperatorType):
    """Represents NewProtocol[*Members]"""
    members: list[Type]

class NewTypedDictType(TypeOperatorType):
    """Represents NewTypedDict[*Members]"""
    members: list[Type]

class IterType(ProperType):
    """Represents Iter[T] - marks a type for iteration"""
    inner: Type
```

#### String Operation Types

```python
class SliceType(TypeOperatorType):
    """Represents Slice[S, Start, End]"""
    target: Type  # Literal string
    start: Type   # Literal int or None
    end: Type     # Literal int or None

class ConcatType(TypeOperatorType):
    """Represents Concat[S1, S2]"""
    left: Type
    right: Type

class StringCaseType(TypeOperatorType):
    """Base for Uppercase, Lowercase, Capitalize, Uncapitalize"""
    target: Type
    operation: str  # 'upper', 'lower', 'capitalize', 'uncapitalize'
```

#### Annotated Operations

```python
class GetAnnotationsType(TypeOperatorType):
    """Represents GetAnnotations[T]"""
    target: Type

class DropAnnotationsType(TypeOperatorType):
    """Represents DropAnnotations[T]"""
    target: Type
```

### 1.2 Update Type Visitor (`mypy/type_visitor.py`)

Add visitor methods for each new type:

```python
class TypeVisitor(Generic[T]):
    # ... existing methods ...

    def visit_member_type(self, t: MemberType) -> T: ...
    def visit_param_type(self, t: ParamType) -> T: ...
    def visit_conditional_type(self, t: ConditionalType) -> T: ...
    def visit_type_for_comprehension(self, t: TypeForComprehension) -> T: ...
    def visit_get_arg_type(self, t: GetArgType) -> T: ...
    def visit_get_args_type(self, t: GetArgsType) -> T: ...
    def visit_get_attr_type(self, t: GetAttrType) -> T: ...
    def visit_members_type(self, t: MembersType) -> T: ...
    def visit_attrs_type(self, t: AttrsType) -> T: ...
    def visit_from_union_type(self, t: FromUnionType) -> T: ...
    def visit_new_protocol_type(self, t: NewProtocolType) -> T: ...
    def visit_new_typed_dict_type(self, t: NewTypedDictType) -> T: ...
    def visit_iter_type(self, t: IterType) -> T: ...
    def visit_slice_type(self, t: SliceType) -> T: ...
    def visit_concat_type(self, t: ConcatType) -> T: ...
    def visit_string_case_type(self, t: StringCaseType) -> T: ...
    def visit_get_annotations_type(self, t: GetAnnotationsType) -> T: ...
    def visit_drop_annotations_type(self, t: DropAnnotationsType) -> T: ...
```

### 1.3 Register Special Form Names (`mypy/types.py`)

Add constants for the new special forms:

```python
TYPE_LEVEL_NAMES: Final = frozenset({
    'typing.Member',
    'typing.Param',
    'typing.Sub',
    'typing.GetArg',
    'typing.GetArgs',
    'typing.GetAttr',
    'typing.GetName',
    'typing.GetType',
    'typing.GetQuals',
    'typing.GetInit',
    'typing.GetDefiner',
    'typing.Members',
    'typing.Attrs',
    'typing.FromUnion',
    'typing.NewProtocol',
    'typing.NewTypedDict',
    'typing.Iter',
    'typing.Slice',
    'typing.Concat',
    'typing.Uppercase',
    'typing.Lowercase',
    'typing.Capitalize',
    'typing.Uncapitalize',
    'typing.GetAnnotations',
    'typing.DropAnnotations',
    'typing.Length',
})
```

---

## Phase 2: Type Analysis (`mypy/typeanal.py`)

### 2.1 Parse New Special Forms

Extend `try_analyze_special_unbound_type()` to handle new constructs:

```python
def try_analyze_special_unbound_type(self, t: UnboundType, fullname: str) -> Type | None:
    # ... existing cases ...

    if fullname == 'typing.Member':
        return self.analyze_member_type(t)
    elif fullname == 'typing.Param':
        return self.analyze_param_type(t)
    elif fullname == 'typing.Sub':
        return self.analyze_sub_condition(t)
    elif fullname == 'typing.GetArg':
        return self.analyze_get_arg(t)
    elif fullname == 'typing.GetArgs':
        return self.analyze_get_args(t)
    elif fullname == 'typing.GetAttr':
        return self.analyze_get_attr(t)
    elif fullname in ('typing.GetName', 'typing.GetType', 'typing.GetQuals',
                      'typing.GetInit', 'typing.GetDefiner'):
        return self.analyze_member_accessor(t, fullname)
    elif fullname == 'typing.Members':
        return self.analyze_members(t)
    elif fullname == 'typing.Attrs':
        return self.analyze_attrs(t)
    elif fullname == 'typing.FromUnion':
        return self.analyze_from_union(t)
    elif fullname == 'typing.NewProtocol':
        return self.analyze_new_protocol(t)
    elif fullname == 'typing.NewTypedDict':
        return self.analyze_new_typed_dict(t)
    elif fullname == 'typing.Iter':
        return self.analyze_iter(t)
    elif fullname == 'typing.Slice':
        return self.analyze_slice(t)
    elif fullname == 'typing.Concat':
        return self.analyze_concat(t)
    elif fullname in ('typing.Uppercase', 'typing.Lowercase',
                      'typing.Capitalize', 'typing.Uncapitalize'):
        return self.analyze_string_case(t, fullname)
    elif fullname == 'typing.GetAnnotations':
        return self.analyze_get_annotations(t)
    elif fullname == 'typing.DropAnnotations':
        return self.analyze_drop_annotations(t)
    elif fullname == 'typing.Length':
        return self.analyze_length(t)

    return None
```

### 2.2 Parse Conditional Type Syntax

Need to handle the `X if Cond else Y` syntax in type contexts. This requires modification to how type expressions are parsed.

**Option A**: Use a special form `Cond[TrueType, Condition, FalseType]`
**Option B**: Extend the parser to handle ternary in type contexts

For now, pursue Option A as it's less invasive:

```python
# typing.Cond[TrueType, Sub[T, Base], FalseType]
def analyze_conditional_type(self, t: UnboundType) -> Type:
    if len(t.args) != 3:
        self.fail('Cond requires 3 arguments', t)
        return AnyType(TypeOfAny.from_error)

    true_type = self.anal_type(t.args[0])
    condition = self.analyze_type_condition(t.args[1])
    false_type = self.anal_type(t.args[2])

    return ConditionalType(condition, true_type, false_type)
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

## Phase 3: Type Evaluation Engine (`mypy/typelevel.py` - new file)

### 3.1 Create Type Evaluator

A new module for evaluating type-level computations:

```python
"""Type-level computation evaluation."""

from mypy.types import (
    Type, ProperType, Instance, TupleType, UnionType, LiteralType,
    TypedDictType, CallableType, NoneType, AnyType, UninhabitedType,
    MemberType, ParamType, ConditionalType, TypeForComprehension,
    GetArgType, GetArgsType, GetAttrType, MembersType, AttrsType,
    FromUnionType, NewProtocolType, NewTypedDictType, IterType,
    SliceType, ConcatType, StringCaseType, GetAnnotationsType,
    DropAnnotationsType, TypeCondition, SubtypeCondition,
)
from mypy.subtypes import is_subtype
from mypy.typeops import get_proper_type

class TypeLevelEvaluator:
    """Evaluates type-level computations to concrete types."""

    def __init__(self, api: SemanticAnalyzerInterface):
        self.api = api

    def evaluate(self, typ: Type) -> Type:
        """Main entry point: evaluate a type to its simplified form."""
        typ = get_proper_type(typ)

        if isinstance(typ, ConditionalType):
            return self.eval_conditional(typ)
        elif isinstance(typ, TypeForComprehension):
            return self.eval_comprehension(typ)
        elif isinstance(typ, GetArgType):
            return self.eval_get_arg(typ)
        elif isinstance(typ, GetArgsType):
            return self.eval_get_args(typ)
        elif isinstance(typ, GetAttrType):
            return self.eval_get_attr(typ)
        elif isinstance(typ, MembersType):
            return self.eval_members(typ)
        elif isinstance(typ, AttrsType):
            return self.eval_attrs(typ)
        elif isinstance(typ, FromUnionType):
            return self.eval_from_union(typ)
        elif isinstance(typ, NewProtocolType):
            return self.eval_new_protocol(typ)
        elif isinstance(typ, NewTypedDictType):
            return self.eval_new_typed_dict(typ)
        elif isinstance(typ, SliceType):
            return self.eval_slice(typ)
        elif isinstance(typ, ConcatType):
            return self.eval_concat(typ)
        elif isinstance(typ, StringCaseType):
            return self.eval_string_case(typ)
        elif isinstance(typ, GetAnnotationsType):
            return self.eval_get_annotations(typ)
        elif isinstance(typ, DropAnnotationsType):
            return self.eval_drop_annotations(typ)

        return typ  # Already a concrete type

    def eval_condition(self, cond: TypeCondition) -> bool | None:
        """Evaluate a type condition. Returns None if undecidable."""
        if isinstance(cond, SubtypeCondition):
            left = self.evaluate(cond.left)
            right = self.evaluate(cond.right)
            # Handle type variables - may be undecidable
            if self.contains_unresolved_typevar(left) or self.contains_unresolved_typevar(right):
                return None
            return is_subtype(left, right)
        elif isinstance(cond, NotCondition):
            inner = self.eval_condition(cond.inner)
            return None if inner is None else not inner
        elif isinstance(cond, AndCondition):
            left = self.eval_condition(cond.left)
            right = self.eval_condition(cond.right)
            if left is False or right is False:
                return False
            if left is None or right is None:
                return None
            return True
        elif isinstance(cond, OrCondition):
            left = self.eval_condition(cond.left)
            right = self.eval_condition(cond.right)
            if left is True or right is True:
                return True
            if left is None or right is None:
                return None
            return False
        return None

    def eval_conditional(self, typ: ConditionalType) -> Type:
        """Evaluate X if Cond else Y"""
        result = self.eval_condition(typ.condition)
        if result is True:
            return self.evaluate(typ.true_type)
        elif result is False:
            return self.evaluate(typ.false_type)
        else:
            # Undecidable - keep as ConditionalType
            return typ

    def eval_get_arg(self, typ: GetArgType) -> Type:
        """Evaluate GetArg[T, Base, Idx]"""
        target = self.evaluate(typ.target)
        base = self.evaluate(typ.base)
        idx = self.evaluate(typ.index)

        target = get_proper_type(target)
        base = get_proper_type(base)

        # Extract index as int
        if not isinstance(idx, LiteralType) or not isinstance(idx.value, int):
            return typ  # Can't evaluate without literal index

        index = idx.value

        if isinstance(target, Instance) and isinstance(base, Instance):
            # Find the type args when target is viewed as base
            args = self.get_type_args_for_base(target, base.type)
            if args is not None and 0 <= index < len(args):
                return args[index]
            return UninhabitedType()  # Never

        return typ  # Can't evaluate

    def eval_get_args(self, typ: GetArgsType) -> Type:
        """Evaluate GetArgs[T, Base] -> tuple of args"""
        target = self.evaluate(typ.target)
        base = self.evaluate(typ.base)

        target = get_proper_type(target)
        base = get_proper_type(base)

        if isinstance(target, Instance) and isinstance(base, Instance):
            args = self.get_type_args_for_base(target, base.type)
            if args is not None:
                return TupleType(list(args), self.api.named_type('builtins.tuple'))
            return UninhabitedType()

        return typ

    def eval_members(self, typ: MembersType) -> Type:
        """Evaluate Members[T] -> tuple of Member types"""
        target = self.evaluate(typ.target)
        target = get_proper_type(target)

        if isinstance(target, Instance):
            members = []
            for name, node in target.type.names.items():
                if node.type is not None:
                    member = MemberType(
                        name=LiteralType(name, self.api.named_type('builtins.str')),
                        member_type=node.type,
                        quals=self.extract_member_quals(node),
                        init=self.extract_member_init(node),
                        definer=Instance(target.type, [])
                    )
                    members.append(member)
            return TupleType(members, self.api.named_type('builtins.tuple'))

        return typ

    def eval_attrs(self, typ: AttrsType) -> Type:
        """Evaluate Attrs[T] -> tuple of Member types (annotated attrs only)"""
        # Similar to members but filters to only annotated attributes
        # (excludes methods, class variables without annotations, etc.)
        pass

    def eval_from_union(self, typ: FromUnionType) -> Type:
        """Evaluate FromUnion[T] -> tuple of union elements"""
        target = self.evaluate(typ.target)
        target = get_proper_type(target)

        if isinstance(target, UnionType):
            return TupleType(list(target.items), self.api.named_type('builtins.tuple'))
        else:
            # Non-union becomes 1-tuple
            return TupleType([target], self.api.named_type('builtins.tuple'))

    def eval_comprehension(self, typ: TypeForComprehension) -> Type:
        """Evaluate *[Expr for x in Iter[T] if Cond]"""
        iter_type = self.evaluate(typ.iter_type)
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
                cond_subst = self.substitute_typevar_in_condition(cond, typ.iter_var, item)
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

    def eval_new_protocol(self, typ: NewProtocolType) -> Type:
        """Evaluate NewProtocol[*Members] -> create a new structural type"""
        evaluated_members = [self.evaluate(m) for m in typ.members]

        # All members must be MemberType
        for m in evaluated_members:
            if not isinstance(get_proper_type(m), MemberType):
                return typ  # Can't evaluate yet

        # Create a new TypeInfo for the protocol
        return self.create_protocol_from_members(evaluated_members)

    def eval_new_typed_dict(self, typ: NewTypedDictType) -> Type:
        """Evaluate NewTypedDict[*Members] -> create a new TypedDict"""
        evaluated_members = [self.evaluate(m) for m in typ.members]

        items = {}
        required_keys = set()

        for m in evaluated_members:
            m = get_proper_type(m)
            if not isinstance(m, MemberType):
                return typ  # Can't evaluate yet

            name = self.extract_literal_string(m.name)
            if name is None:
                return typ

            items[name] = m.member_type
            # Check quals for Required/NotRequired
            if not self.has_not_required_qual(m.quals):
                required_keys.add(name)

        return TypedDictType(
            items=items,
            required_keys=required_keys,
            readonly_keys=frozenset(),
            fallback=self.api.named_type('typing.TypedDict')
        )

    # String operations
    def eval_slice(self, typ: SliceType) -> Type:
        """Evaluate Slice[S, Start, End]"""
        target = self.evaluate(typ.target)
        start = self.evaluate(typ.start)
        end = self.evaluate(typ.end)

        s = self.extract_literal_string(target)
        start_val = self.extract_literal_int_or_none(start)
        end_val = self.extract_literal_int_or_none(end)

        if s is not None and start_val is not ... and end_val is not ...:
            result = s[start_val:end_val]
            return LiteralType(result, self.api.named_type('builtins.str'))

        return typ

    def eval_concat(self, typ: ConcatType) -> Type:
        """Evaluate Concat[S1, S2]"""
        left = self.extract_literal_string(self.evaluate(typ.left))
        right = self.extract_literal_string(self.evaluate(typ.right))

        if left is not None and right is not None:
            return LiteralType(left + right, self.api.named_type('builtins.str'))

        return typ

    def eval_string_case(self, typ: StringCaseType) -> Type:
        """Evaluate Uppercase, Lowercase, Capitalize, Uncapitalize"""
        target = self.extract_literal_string(self.evaluate(typ.target))

        if target is not None:
            if typ.operation == 'upper':
                result = target.upper()
            elif typ.operation == 'lower':
                result = target.lower()
            elif typ.operation == 'capitalize':
                result = target.capitalize()
            elif typ.operation == 'uncapitalize':
                result = target[0].lower() + target[1:] if target else target
            else:
                return typ
            return LiteralType(result, self.api.named_type('builtins.str'))

        return typ

    # Helper methods
    def get_type_args_for_base(self, instance: Instance, base: TypeInfo) -> list[Type] | None:
        """Get type args when viewing instance as base class."""
        # Walk MRO to find base and map type arguments
        pass

    def contains_unresolved_typevar(self, typ: Type) -> bool:
        """Check if type contains unresolved type variables."""
        pass

    def substitute_typevar(self, typ: Type, var_name: str, replacement: Type) -> Type:
        """Substitute a type variable with a concrete type."""
        pass

    def extract_literal_string(self, typ: Type) -> str | None:
        """Extract string value from LiteralType."""
        typ = get_proper_type(typ)
        if isinstance(typ, LiteralType) and isinstance(typ.value, str):
            return typ.value
        return None

    def extract_literal_int_or_none(self, typ: Type) -> int | None | ...:
        """Extract int or None from LiteralType. Returns ... if not extractable."""
        typ = get_proper_type(typ)
        if isinstance(typ, NoneType):
            return None
        if isinstance(typ, LiteralType) and isinstance(typ.value, int):
            return typ.value
        return ...  # sentinel for "not extractable"

    def create_protocol_from_members(self, members: list[Type]) -> Type:
        """Create a new Protocol TypeInfo from Member types."""
        # This needs to create synthetic TypeInfo
        pass
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

    def visit_conditional_type(self, t: ConditionalType) -> Type:
        return ConditionalType(
            self.expand_condition(t.condition),
            t.true_type.accept(self),
            t.false_type.accept(self),
        )

    def visit_type_for_comprehension(self, t: TypeForComprehension) -> Type:
        # Don't substitute the iteration variable
        return TypeForComprehension(
            t.element_expr.accept(self),
            t.iter_var,
            t.iter_type.accept(self),
            [self.expand_condition(c) for c in t.conditions],
        )

    # ... more visit methods for other new types ...
```

### 4.3 Subtype Checking (`mypy/subtypes.py`)

Add subtype rules for new types:

```python
class SubtypeVisitor(TypeVisitor[bool]):
    # ... existing methods ...

    def visit_conditional_type(self, left: ConditionalType) -> bool:
        # A conditional type is subtype if both branches are subtypes
        # OR if we can evaluate the condition
        evaluator = TypeLevelEvaluator(...)
        result = evaluator.eval_condition(left.condition)

        if result is True:
            return is_subtype(left.true_type, self.right)
        elif result is False:
            return is_subtype(left.false_type, self.right)
        else:
            # Must be subtype in both cases
            return (is_subtype(left.true_type, self.right) and
                    is_subtype(left.false_type, self.right))
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

### Milestone 1: Core Type Operators (Weeks 1-3)
1. Add `MemberType`, `ParamType` type classes
2. Add `GetArg`, `GetArgs`, `FromUnion` operators
3. Add `Members`, `Attrs` operators
4. Basic type evaluator for these operators
5. Tests for basic operations

### Milestone 2: Conditional Types (Weeks 4-5)
1. Add `ConditionalType` and condition classes
2. Add `Sub` condition operator
3. Integrate with type evaluator
4. Tests for conditionals

### Milestone 3: Type Comprehensions (Weeks 6-7)
1. Add `TypeForComprehension`, `IterType`
2. Parser support for comprehension syntax
3. Evaluator support for comprehensions
4. Tests for comprehensions

### Milestone 4: NewProtocol/NewTypedDict (Weeks 8-10)
1. Add `NewProtocolType`, `NewTypedDictType`
2. Implement synthetic TypeInfo creation
3. Integration with type checking
4. Tests for type construction

### Milestone 5: Extended Callables (Weeks 11-12)
1. Full `Param` type support
2. Callable introspection
3. Extended callable construction
4. Tests for callables

### Milestone 6: String Operations (Week 13)
1. Add string operation types
2. Implement string evaluators
3. Tests for string ops

### Milestone 7: Annotated & InitField (Weeks 14-15)
1. Preserve Annotated metadata
2. GetAnnotations/DropAnnotations
3. InitField support
4. Tests

### Milestone 8: TypedDict kwargs inference (Week 16)
1. `Unpack[K]` for TypeVar K
2. Inference from kwargs
3. Tests

### Milestone 9: Integration & Polish (Weeks 17-20)
1. Full PEP examples working
2. Error messages
3. Documentation
4. Performance optimization

---

## Key Design Decisions

### 1. Lazy vs Eager Evaluation
**Decision**: Lazy evaluation with caching. Type-level computations are evaluated when needed (e.g., during subtype checking) rather than immediately during parsing.

### 2. Handling Undecidable Conditions
**Decision**: When a condition cannot be evaluated (e.g., involves unbound type variables), preserve the conditional type. It will be evaluated later when more type information is available.

### 3. Synthetic Type Identity
**Decision**: Types created via `NewProtocol` are structural (protocols), so identity is based on structure, not name. Each creation point may produce a "different" type that is structurally equivalent.

### 4. Error Handling
**Decision**: Invalid type-level operations (e.g., `GetArg` on non-generic type) return `Never` rather than raising errors, consistent with the spec.

### 5. Runtime Evaluation
**Decision**: This implementation focuses on static type checking. Runtime evaluation is a separate library concern (as noted in the spec).

---

## Files to Create/Modify

### New Files
- `mypy/typelevel.py` - Type-level computation evaluator
- `mypy/test/test_typelevel_*.py` - Test files
- `test-data/unit/check-typelevel-*.test` - Test data

### Modified Files
- `mypy/types.py` - Add new type classes
- `mypy/type_visitor.py` - Add visitor methods
- `mypy/typeanal.py` - Parse new special forms
- `mypy/expandtype.py` - Expand new types
- `mypy/subtypes.py` - Subtype rules
- `mypy/checkexpr.py` - kwargs inference
- `mypy/semanal.py` - InitField handling
- `mypy/nodes.py` - Possibly extend TypeInfo

---

## Open Questions for Discussion

1. **Syntax for conditionals**: Use `X if Sub[T, Base] else Y` (requires parser changes) or `Cond[X, Sub[T, Base], Y]` (works with existing syntax)?

2. **Protocol vs TypedDict creation**: Should `NewProtocol` create true protocols (with `is_protocol=True`) or just structural types?

3. **Type alias recursion**: How to handle recursive type aliases that use type-level computation?

4. **Error recovery**: What should happen when type-level computation fails? Currently spec says return `Never`.

5. **Caching strategy**: How aggressively to cache evaluated type-level computations?

6. **Plugin interaction**: Should plugins be able to define custom type operators?
