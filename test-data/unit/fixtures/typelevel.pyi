# Builtins stub used in typelevel-related test cases.

import _typeshed
from typing import Iterable, Iterator, TypeVar, Generic, Sequence, Mapping, Optional, overload, Tuple, Type, Union, Self, type_check_only, _type_operator

_T = TypeVar("_T")
_Tco = TypeVar('_Tco', covariant=True)

class object:
    def __init__(self) -> None: pass
    def __new__(cls) -> Self: ...
    def __str__(self) -> str: pass

class type:
    def __init__(self, *a: object) -> None: pass
    def __call__(self, *a: object) -> object: pass
class tuple(Sequence[_Tco], Generic[_Tco]):
    def __hash__(self) -> int: ...
    def __new__(cls: Type[_T], iterable: Iterable[_Tco] = ...) -> _T: ...
    def __iter__(self) -> Iterator[_Tco]: pass
    def __contains__(self, item: object) -> bool: pass
    @overload
    def __getitem__(self, x: int) -> _Tco: pass
    @overload
    def __getitem__(self, x: slice) -> Tuple[_Tco, ...]: ...
    def __mul__(self, n: int) -> Tuple[_Tco, ...]: pass
    def __rmul__(self, n: int) -> Tuple[_Tco, ...]: pass
    def __add__(self, x: Tuple[_Tco, ...]) -> Tuple[_Tco, ...]: pass
    def count(self, obj: object) -> int: pass
class function:
    __name__: str
class ellipsis: pass
class classmethod: pass

# We need int and slice for indexing tuples.
class int:
    def __neg__(self) -> 'int': pass
    def __pos__(self) -> 'int': pass
class float: pass
class slice: pass
class bool(int): pass
class str: pass # For convenience
class bytes: pass
class bytearray: pass

class list(Sequence[_T], Generic[_T]):
    @overload
    def __getitem__(self, i: int) -> _T: ...
    @overload
    def __getitem__(self, s: slice) -> list[_T]: ...
    def __contains__(self, item: object) -> bool: ...
    def __iter__(self) -> Iterator[_T]: ...

def isinstance(x: object, t: type) -> bool: pass

class BaseException: pass

KT = TypeVar('KT')
VT = TypeVar('VT')
T = TypeVar('T')

class dict(Mapping[KT, VT]):
    @overload
    def __init__(self, **kwargs: VT) -> None: pass
    @overload
    def __init__(self, arg: Iterable[Tuple[KT, VT]], **kwargs: VT) -> None: pass
    def __getitem__(self, key: KT) -> VT: pass
    def __setitem__(self, k: KT, v: VT) -> None: pass
    def __iter__(self) -> Iterator[KT]: pass
    def __contains__(self, item: object) -> int: pass
    @overload
    def get(self, k: KT) -> Optional[VT]: pass
    @overload
    def get(self, k: KT, default: Union[VT, T]) -> Union[VT, T]: pass
    def __len__(self) -> int: ...


# Type-level computation stuff

_TrueType = TypeVar('_TrueType')
_FalseType = TypeVar('_FalseType')

@type_check_only
@_type_operator
class _Cond(Generic[_T, _TrueType, _FalseType]): ...

_T2 = TypeVar('_T2')

@type_check_only
@_type_operator
class _And(Generic[_T, _T2]): ...

@type_check_only
@_type_operator
class _Or(Generic[_T, _T2]): ...

@type_check_only
@_type_operator
class _Not(Generic[_T]): ...

@type_check_only
@_type_operator
class _DictEntry(Generic[_T, _T2]): ...

@type_check_only
@_type_operator
class _TypeGetAttr(Generic[_T, _T2]): ...
