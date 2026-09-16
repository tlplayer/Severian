# Severian: Manual
# Author: Timothy Player
# Date 2026-09-14
# Status: Draft

## Introduction

Severian is a programming language that deals with these cyclical problems:
- Code should have quality through tests, naming, standards, types, and simplicity
- Code should be fast and flexible
- Rewriting code is a waste, extending code and improving one place should apply broadly to many places


## Prelude

The prelude is what every sev function gets without imports

```
my_list = []

```

## Classes

Classes attempt to do a couple things, standardize set,get methods. 

'''
Objects expose generic get() and set() operations to avoid getX()/setX()
methods for every field.

Field constraints may return Errors directly or delegate to functions.
Builders use the same set() path, so construction and later mutation obey
the same validation rules.

Tests use:
    assert(...)      hard requirement
    expect(...)      record failure and continue
    throws(...)      expected Error
    mock(...)        test-scoped function behavior
'''


## Testing

Tests are seperated cleanly from code. this provides core benefits. 
- Compilation can be isolated from testing.
- No external libraries are needed to bake into the project
- Function naming can be a string to avoid test_foo_bar() notation which becomes word soup
- different imports can live solely in the test


```sev
test "Foo is a valid function":

```