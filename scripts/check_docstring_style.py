"""Check Sphinx field syntax, following the SynKit docstring convention.

Short prose docstrings are valid. Fields, when present, must match the signature.
"""
from __future__ import annotations

import argparse
import ast
import inspect
import re

from check_python_file_size import DOC_OWNERS, python_files

HEADINGS = {
    "Args", "Arguments", "Parameters", "Keyword Args", "Keyword Arguments", "Other Parameters",
    "Returns", "Yields", "Raises", "Attributes", "Public Properties", "Methods", "Notes", "Examples",
    "Warnings", "See Also", "References", "Example", "Note", "Warning",
}


def docstring_errors(tree: ast.AST) -> list[tuple[int, str]]:
    errors = []
    for node in ast.walk(tree):
        if not isinstance(node, DOC_OWNERS) or ast.get_docstring(node) is None:
            continue
        value = node.body[0].value
        lines = inspect.cleandoc(value.value).splitlines()
        first_line = next((i for i, line in enumerate(value.value.splitlines()) if line.strip()), 0)
        parameters = None
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = node.args
            parameters = {arg.arg for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs)}
            parameters.update(arg.arg for arg in (args.vararg, args.kwarg) if arg is not None)
        for index, line in enumerate(lines):
            message = None
            stripped = line.strip()
            next_line = lines[index + 1].strip() if index + 1 < len(lines) else ""
            heading = stripped.removesuffix(":")
            if heading in HEADINGS and (stripped.endswith(":") or re.fullmatch(r"-{3,}", next_line)):
                message = "use Sphinx fields or prose instead of section headings"
            if re.match(r":(?:returns|yields|raises|Examples?|examples?):", stripped):
                message = "use :return:, :yield:, :raises Exception: or a literal example block"
            field = re.match(r":(?:param|type)\s+([^:]+):", stripped)
            if field:
                name = field[1].strip()
                if len(name.split()) != 1:
                    message = "put parameter types in a separate :type name: field"
                elif parameters is not None and name.lstrip("*") not in parameters:
                    message = f"documented parameter {name!r} is absent from the signature"
            if re.match(r":(?:param|type|return|rtype|yield|ytype|raises)\b", stripped) and line != line.lstrip():
                message = "Sphinx fields must start at the docstring margin"
            if message:
                errors.append((value.lineno + first_line + index, message))
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", default=["synrxn", "Test", "scripts"])
    args = parser.parse_args()
    count = 0
    try:
        for path in python_files(args.paths):
            source = path.read_text(encoding="utf-8")
            for line, message in docstring_errors(ast.parse(source, filename=str(path))):
                print(f"{path}:{line}: {message}")
                count += 1
    except (ValueError, SyntaxError) as error:
        print(error)
        return 1
    if count:
        print(f"{count} docstring style findings.")
    else:
        print("Docstrings follow the Sphinx field convention.")
    return int(bool(count))


if __name__ == "__main__":
    raise SystemExit(main())
