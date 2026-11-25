"""Small provenance and I/O helpers for maintained reconstruction modules.

Active implementations contain ordinary Python bodies. Historical originals
live outside the package and are read only for explicit source-history tools,
source migration, or checksummed provenance and native reconstruction.
"""
from __future__ import annotations

import ast
from functools import wraps
import os
from pathlib import Path
from types import FunctionType

from synrxn.reconstruction.paths import HistoricalPath, REPO, legacy_path, source_path


class _HistoricalImports(ast.NodeTransformer):
    """Resolve the old script-local processing imports inside the package."""

    def visit_ImportFrom(self, node):
        if node.module == 'pathlib' and node.level == 0:
            # Only byte-frozen kernels receive logical display paths with
            # alias-aware I/O. Active package code uses explicit path helpers.
            ordinary = [alias for alias in node.names if alias.name != 'Path']
            redirected = [ast.alias(name='HistoricalPath', asname=alias.asname or 'Path')
                          for alias in node.names if alias.name == 'Path']
            statements = []
            if ordinary:
                statements.append(ast.ImportFrom(module='pathlib', names=ordinary, level=0))
            if redirected:
                statements.append(ast.ImportFrom(module='synrxn.reconstruction.paths', names=redirected, level=0))
            return [ast.copy_location(item, node) for item in statements]
        if node.level == 0 and node.module and (node.module == 'kdd.coverage' or node.module.startswith('kdd.coverage.')):
            node.module = 'synrxn.reconstruction.coverage' + node.module.removeprefix('kdd.coverage')
        elif node.module in {'kdd.reproduce', 'kdd.prepare_ncs_yield_splits'} and node.level == 0:
            node.module = {'kdd.reproduce': 'synrxn.reconstruction.families.suite',
                           'kdd.prepare_ncs_yield_splits': 'synrxn.reconstruction.families.outcomes.reaction_success._bh_hte_ood'}[node.module]
        elif node.module == 'kdd' and node.level == 0:
            names = {'reproduce': 'suite', 'prepare_ncs_yield_splits': 'assay'}
            if any(alias.name not in names for alias in node.names):
                raise ValueError('Historical package import requires an explicit package facade')
            node.module = 'synrxn.reconstruction'
            node.names = [ast.alias(name=names[alias.name], asname=alias.asname or alias.name)
                          for alias in node.names]
        elif node.module in {'process.aam', 'process.rbl'} and node.level == 0:
            task = {'process.aam': 'atom_mapping', 'process.rbl': 'reaction_balancing'}[node.module]
            node.module = f'synrxn.reconstruction.families.representation.{task}._sources.conversion'
        elif node.module in {'script.build_classification_dataset', 'script.build_synthesis_dataset'} and node.level == 0:
            node.module = {'script.build_classification_dataset': 'synrxn.reconstruction.families.representation.reaction_classification._classification',
                           'script.build_synthesis_dataset': 'synrxn.reconstruction.families.transformations._synthesis'}[node.module]
        elif node.module == 'script.audit_ncs_baselines' and node.level == 0:
            statements = ast.parse('from synrxn.reconstruction.provenance.source_history import load_tool as __load_tool\n'
                                   '__old_tool = __load_tool("audit_ncs_baselines")').body
            for alias in node.names:
                if alias.name == '*':
                    raise ValueError('Historical wildcard imports require an explicit package facade')
                statements.append(ast.Assign(targets=[ast.Name(id=alias.asname or alias.name, ctx=ast.Store())],
                                              value=ast.Attribute(value=ast.Name(id='__old_tool', ctx=ast.Load()),
                                                                  attr=alias.name, ctx=ast.Load())))
            return [ast.copy_location(item, node) for item in statements]
        return node

    def visit_keyword(self, node):
        if node.arg == 'default':
            value = node.value
            if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == 'Path' and value.args:
                value = value.args[0]
            if isinstance(value, ast.Constant) and isinstance(value.value, str) and value.value.startswith('reports/'):
                value.value = '.cache/synrxn/runs/builders/' + value.value.removeprefix('reports/')
        return self.generic_visit(node)

    def visit_Call(self, node):
        self.generic_visit(node)
        # Two original property-curation sinks explicitly stringify their
        # filename, bypassing os.fspath(). Keep display str() logical elsewhere.
        if (isinstance(node.func, ast.Attribute) and node.func.attr in {'read_csv', 'to_csv'}
                and node.args and isinstance(node.args[0], ast.Call)
                and isinstance(node.args[0].func, ast.Name) and node.args[0].func.id == 'str'
                and len(node.args[0].args) == 1):
            node.args[0] = ast.copy_location(ast.Call(
                func=ast.Name(id='__synrxn_io_path', ctx=ast.Load()),
                args=node.args[0].args, keywords=[]), node.args[0])
        return node


def _mcsa_asset_paths(tree):
    """Keep original size and digest checks on explicitly relocated sources."""
    for function in tree.body:
        if isinstance(function, ast.FunctionDef) and function.name == 'verify_assets':
            for node in ast.walk(function):
                if (isinstance(node, ast.Assign) and isinstance(node.value, ast.BinOp)
                        and isinstance(node.value.op, ast.Div)
                        and isinstance(node.value.left, ast.Name) and node.value.left.id == 'repo'
                        and isinstance(node.value.right, ast.Name) and node.value.right.id == 'relative'):
                    node.value = ast.copy_location(ast.Call(
                        func=ast.Name(id='__synrxn_source_path', ctx=ast.Load()),
                        args=[ast.Name(id='relative', ctx=ast.Load())],
                        keywords=[ast.keyword(arg='repo', value=ast.Name(id='repo', ctx=ast.Load()))]),
                        node.value)
                    return
    raise ValueError('Frozen M-CSA asset verifier no longer matches its path contract')


class _SourceCopies:
    """Copy the frozen importer when a historical evidence snapshot requests it."""

    def __init__(self, module):
        self.module = module

    def __getattr__(self, name):
        return getattr(self.module, name)

    def copyfile(self, source, destination, *args, **kwargs):
        return self.module.copyfile(source_path(source), destination, *args, **kwargs)


class _PackagedMcsaSubprocess:
    """Redirect only the retired M-CSA adapter invocation to its package CLI."""

    def __init__(self, module):
        self.module = module

    def __getattr__(self, name):
        return getattr(self.module, name)

    def run(self, command, *args, **kwargs):
        if (isinstance(command, (list, tuple)) and len(command) >= 2
                and str(command[1]) == str(REPO / 'script/prepare_mcsa_task.py')):
            command = [command[0], '-m', 'synrxn.reconstruction.families.mechanisms.mechanistic_prediction._mcsa', *command[2:]]
        return self.module.run(command, *args, **kwargs)


def _quantumpioneer_fetch_import(tree):
    """Use the packaged downloader instead of importing a retired root file."""
    for function in tree.body:
        if isinstance(function, ast.FunctionDef) and function.name == 'fetch':
            rewritten = []
            matches = 0
            for statement in function.body:
                if (isinstance(statement, ast.Assign) and len(statement.targets) == 1
                        and isinstance(statement.targets[0], ast.Name)
                        and statement.targets[0].id == 'spec'):
                    rewritten += ast.parse('from synrxn.reconstruction.expansion import fetch as module').body
                    matches += 1
                elif (isinstance(statement, ast.Assign) and len(statement.targets) == 1
                      and isinstance(statement.targets[0], ast.Name)
                      and statement.targets[0].id == 'module'):
                    matches += 1
                elif (isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call)
                      and isinstance(statement.value.func, ast.Attribute)
                      and statement.value.func.attr == 'exec_module'):
                    matches += 1
                else:
                    rewritten.append(statement)
            if matches != 3:
                raise ValueError('Frozen QuantumPioneer downloader no longer matches its import contract')
            function.body = rewritten
            return
    raise ValueError('Frozen QuantumPioneer downloader has no fetch function')


def _main_guard(node):
    return (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name) and node.test.left.id == '__name__'
            and len(node.test.comparators) == 1 and isinstance(node.test.comparators[0], ast.Constant)
            and node.test.comparators[0].value == '__main__')


def _historical_arguments(function):
    """Keep Path arguments supplied by active callers in the kernel's scope."""
    def convert(value):
        return HistoricalPath(str(value)) if isinstance(value, Path) else value

    @wraps(function)
    def call(*args, **kwargs):
        return function(*(convert(value) for value in args),
                        **{name: convert(value) for name, value in kwargs.items()})
    return call


def _io_path(value):
    """Resolve a filename at explicitly identified stringifying I/O sinks."""
    if isinstance(value, str) and '://' in value:
        return value
    return os.fspath(HistoricalPath(value))


class _ExplicitSourceFile(ast.NodeTransformer):
    """Keep module identity honest while retaining historical source locators."""

    def visit_Name(self, node):
        if node.id == '__file__' and isinstance(node.ctx, ast.Load):
            return ast.copy_location(ast.Name(id='__source_file__', ctx=ast.Load()), node)
        return node


def _prepared_tree(payload, relative, filename):
    tree = ast.parse(payload, filename=str(filename))
    tree = _HistoricalImports().visit(tree)
    tree = _ExplicitSourceFile().visit(tree)
    if relative == 'kdd/coverage/adapters/quantumpioneer.py':
        _quantumpioneer_fetch_import(tree)
    if relative == 'kdd/coverage/adapters/mcsa.py':
        _mcsa_asset_paths(tree)
    return ast.fix_missing_locations(tree)


def configure_runtime(namespace, relative):
    """Configure direct package implementations without loading archived code.

    The historical source locator is a provenance identity, separate from the
    module's real __file__. Only checksum helpers consult a preserved original
    when asked to verify its historical source pin.
    """
    namespace['__synrxn_io_path'] = _io_path
    namespace['__synrxn_source_path'] = source_path
    # Imported functions keep their own pathlib semantics. Only definitions
    # belonging to this module receive its historical provenance-path scope.
    for name, value in list(namespace.items()):
        if isinstance(value, FunctionType) and value.__globals__ is namespace:
            namespace[name] = _historical_arguments(value)
    if relative == 'script/prepare_synepd_task.py':
        namespace['shutil'] = _SourceCopies(namespace['shutil'])
    if relative == 'script/rebuild_mcsa.py':
        namespace['subprocess'] = _PackagedMcsaSubprocess(namespace['subprocess'])
    for name in ['sha', 'sha256', 'digest']:
        original = namespace.get(name)
        if callable(original):
            @wraps(original)
            def aliased(path, *args, _original=original, **kwargs):
                try:
                    logical = Path(path)
                    logical = logical if logical.is_absolute() else REPO / logical
                    candidate = source_path(path)
                    # Some kernels call a string-hashing function ``digest``.
                    # Only substitute an explicitly archived source alias;
                    # preserve ordinary values, paths and argument types.
                    resolved = candidate if candidate != logical else path
                    if (logical == REPO / relative and not candidate.is_file()
                            and namespace.get('__file__')):
                        # An installed wheel has maintained implementations,
                        # not the research source archive. Hash its actual
                        # source when reporting this module's own code; never
                        # substitute an unverified cached historical digest.
                        current = Path(namespace['__file__'])
                        if current.is_file():
                            resolved = current
                except (OSError, TypeError, ValueError):
                    resolved = path
                return _original(resolved, *args, **kwargs)
            namespace[name] = aliased
    if relative == 'kdd/coverage/rights.py' and '_path' in namespace:
        original_path = namespace['_path']
        def aliased_rights_path(repo, value):
            path = original_path(repo, value)
            return source_path(path, repo) if path is not None else None
        namespace['_path'] = aliased_rights_path


def materialize_source(relative, original_path=None):
    """Return an editable, direct module body for an explicit original source.

    Used once for repository organization, never during active module imports.
    Original source bytes remain in archive/source-code; this emits ordinary
    maintained Python with explicit imports and a small runtime configuration.
    """
    from synrxn.reconstruction.paths import historical_file
    historical_file(relative)  # Validate the logical identity before emitting code.
    resource = Path(original_path) if original_path is not None else legacy_path(relative)
    original = resource.read_bytes()
    tree = _prepared_tree(original, relative, resource)
    guards = [node for node in tree.body if _main_guard(node)]
    definitions = [node for node in tree.body if not _main_guard(node)]
    position = 0
    if definitions and isinstance(definitions[0], ast.Expr) and isinstance(definitions[0].value, ast.Constant) and isinstance(definitions[0].value.value, str):
        position = 1
    while position < len(definitions) and isinstance(definitions[position], ast.ImportFrom) and definitions[position].module == '__future__':
        position += 1
    used_names = {node.id for node in ast.walk(tree)
                  if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)}
    path_imports = ['historical_file as __historical_file']
    runtime_imports = ['configure_runtime as __configure_runtime']
    if '__synrxn_source_path' in used_names:
        path_imports.append('source_path as __synrxn_source_path')
    if '__synrxn_io_path' in used_names:
        runtime_imports.append('_io_path as __synrxn_io_path')
    header = ast.parse(
        'from synrxn.reconstruction.paths import ' + ', '.join(path_imports) + '\n'
        'from synrxn.reconstruction.provenance.compat import ' + ', '.join(runtime_imports) + '\n'
        '__source_file__ = __historical_file(' + repr(relative) + ')\n').body
    configure = ast.parse('__configure_runtime(globals(), ' + repr(relative) + ')').body
    tree = ast.Module(body=[*definitions[:position], *header, *definitions[position:], *configure, *guards], type_ignores=[])
    ast.fix_missing_locations(tree)
    # Compiling here catches invalid future-import ordering or transformed AST
    # before a caller replaces the current package implementation.
    compile(tree, str(resource), 'exec')
    notices = [line for line in original.decode('utf8').splitlines()
               if line.lstrip().startswith('#') and any(
                   marker in line.casefold() for marker in ('spdx-license', 'copyright', 'license:'))]
    prefix = '# Direct implementation; historical source identity is explicit below.\n'
    if notices:
        prefix += '\n'.join(notices) + '\n'
    return prefix + ast.unparse(tree) + '\n'


def load_legacy_module(namespace, relative):
    """Explicitly inspect or execute an archived historical utility.

    Active dataset builders contain their own implementation and do not use
    this archival loader. Original bytes are required only for this explicit
    operation and for pinned provenance/source-reproduction checks.
    """
    from synrxn.reconstruction.paths import historical_file
    resource = legacy_path(relative)
    tree = _prepared_tree(resource.read_bytes(), relative, resource)
    guards = [node for node in tree.body if _main_guard(node)]
    definitions = ast.Module(body=[node for node in tree.body if not _main_guard(node)], type_ignores=[])
    namespace['__file__'] = str(resource)
    namespace['__source_file__'] = historical_file(relative)
    namespace['__synrxn_io_path'] = _io_path
    namespace['__synrxn_source_path'] = source_path
    exec(compile(definitions, str(resource), 'exec'), namespace)
    configure_runtime(namespace, relative)
    if namespace.get('__name__') == '__main__':
        for guard in guards:
            exec(compile(ast.Module(body=guard.body, type_ignores=[]), str(resource), 'exec'), namespace)
