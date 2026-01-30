This file provides guidance to coding agents, I guess.
Also to humans some probably.

## Current Work: Typemap PEP Implementation

We are implementing a PEP draft for type-level computation. The specification
is in `pep.rst`. Refer to it when implementing new type operators
or features.

## Default virtualenv

The default virtualenv is ``venv``, so most of the commands below
should be run from ``venv/bin`` if available.


## Pre-commit

Always run ``venv/bin/python runtests.py lint self`` before committing
and make sure that it passes.


## Common Commands

### Running Tests
```bash
# Run a single test by name (uses pytest -k matching)
pytest -n0 -k testNewSyntaxBasics

# Run all tests in a specific test file
pytest mypy/test/testcheck.py::TypeCheckSuite::check-dataclasses.test

# Run tests matching a pattern
pytest -q -k "MethodCall"

# Run the full test suite (slow)
python runtests.py

# Run with debugging (disables parallelization)
pytest -n0 --pdb -k testName
```

### Linting and Type Checking
```bash
# Run formatters and linters
python runtests.py lint

# Type check mypy's own code
python -m mypy --config-file mypy_self_check.ini -p mypy
```

### Manual Testing
```bash
# Run mypy directly on a file
python -m mypy PROGRAM.py

# Run mypy on a module
python -m mypy -m MODULE
```

## Architecture Overview

Mypy is a static type checker that processes Python code through multiple passes:

### Core Pipeline
1. **Parsing** (`fastparse.py`) - Converts source to AST using Python's `ast` module
2. **Semantic Analysis** (`semanal.py`, `semanal_main.py`) - Resolves names, builds symbol tables, analyzes imports
3. **Type Checking** (`checker.py`, `checkexpr.py`) - Verifies type correctness

### Key Data Structures

**AST Nodes** (`nodes.py`):
- `MypyFile` - Root of a parsed module
- `FuncDef`, `ClassDef` - Function/class definitions
- `TypeInfo` - Metadata about classes (bases, MRO, members)
- `SymbolTable`, `SymbolTableNode` - Name resolution

**Types** (`types.py`):
- `Type` - Base class for all types
- `ProperType` - Concrete types (Instance, CallableType, TupleType, UnionType, etc.)
- `TypeAliasType` - Type aliases that expand to proper types
- `get_proper_type()` - Expands type aliases to proper types

### Type Operations
- `subtypes.py` - Subtype checking (`is_subtype()`)
- `meet.py`, `join.py` - Type meets (intersection) and joins (union)
- `expandtype.py` - Type variable substitution
- `typeops.py` - Type utilities and transformations

### Build System
- `build.py` - Orchestrates the entire type checking process
- `State` - Represents a module being processed
- Handles incremental checking and caching

## Test Data Format

Tests in `test-data/unit/check-*.test` use a declarative format:
```
[case testName]
# flags: --some-flag
x: int = "wrong"  # E: Incompatible types...

[builtins fixtures/tuple.pyi]
[typing fixtures/typing-full.pyi]
```

- `# E:` marks expected errors, `# N:` for notes, `# W:` for warnings
- `[builtins fixtures/...]` specifies stub files for builtins
- `[typing fixtures/typing-full.pyi]` uses extended typing stubs
- Tests use minimal stubs by default; define needed classes in test or use fixtures
