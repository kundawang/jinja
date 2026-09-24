import itertools
import typing as t

from . import nodes
from .visitor import NodeVisitor

if t.TYPE_CHECKING:
    import typing_extensions as te

VAR_LOAD_PARAMETER = "param"
VAR_LOAD_RESOLVE = "resolve"
VAR_LOAD_ALIAS = "alias"
VAR_LOAD_UNDEFINED = "undefined"


def find_symbols(
    nodes: t.Iterable[nodes.Node], parent_symbols: t.Optional["Symbols"] = None
) -> "Symbols":
    sym = Symbols(parent=parent_symbols)
    visitor = FrameSymbolVisitor(sym)
    for node in nodes:
        visitor.visit(node)
    return sym


def symbols_for_node(
    node: nodes.Node, parent_symbols: t.Optional["Symbols"] = None
) -> "Symbols":
    sym = Symbols(parent=parent_symbols)
    sym.analyze_node(node)
    return sym


def _defines_local(symbols: "Symbols", name: str) -> bool:
    """Whether ``symbols`` introduced ``name`` as a new local, assigning it
    before any read.

    If the name was read first, or an outer scope already defines it, the
    outer value is still needed and must be loaded. See issue :issue:`1253`.
    """
    ref = symbols.refs.get(name)
    return ref is not None and symbols.loads.get(ref) == (VAR_LOAD_UNDEFINED, None)


class Symbols:
    def __init__(
        self, parent: t.Optional["Symbols"] = None, level: int | None = None
    ) -> None:
        if level is None:
            if parent is None:
                level = 0
            else:
                level = parent.level + 1

        self.level: int = level
        self.parent = parent
        self.refs: dict[str, str] = {}
        self.loads: dict[str, t.Any] = {}
        self.stores: set[str] = set()

    def analyze_node(self, node: nodes.Node, **kwargs: t.Any) -> None:
        visitor = RootVisitor(self)
        visitor.visit(node, **kwargs)

    def _define_ref(self, name: str, load: tuple[str, str | None] | None = None) -> str:
        ident = f"l_{self.level}_{name}"
        self.refs[name] = ident
        if load is not None:
            self.loads[ident] = load
        return ident

    def find_load(self, target: str) -> t.Any | None:
        if target in self.loads:
            return self.loads[target]

        if self.parent is not None:
            return self.parent.find_load(target)

        return None

    def find_ref(self, name: str) -> str | None:
        if name in self.refs:
            return self.refs[name]

        if self.parent is not None:
            return self.parent.find_ref(name)

        return None

    def ref(self, name: str) -> str:
        rv = self.find_ref(name)
        if rv is None:
            raise AssertionError(
                "Tried to resolve a name to a reference that was"
                f" unknown to the frame ({name!r})"
            )
        return rv

    def copy(self) -> "te.Self":
        rv = object.__new__(self.__class__)
        rv.__dict__.update(self.__dict__)
        rv.refs = self.refs.copy()
        rv.loads = self.loads.copy()
        rv.stores = self.stores.copy()
        return rv

    def store(self, name: str) -> None:
        self.stores.add(name)

        # If we have not see the name referenced yet, we need to figure
        # out what to set it to.
        if name not in self.refs:
            # If there is a parent scope we check if the name has a
            # reference there.  If it does it means we might have to alias
            # to a variable there.
            if self.parent is not None:
                outer_ref = self.parent.find_ref(name)
                if outer_ref is not None:
                    self._define_ref(name, load=(VAR_LOAD_ALIAS, outer_ref))
                    return

            # Otherwise we can just set it to undefined.
            self._define_ref(name, load=(VAR_LOAD_UNDEFINED, None))

    def declare_parameter(self, name: str) -> str:
        self.stores.add(name)
        return self._define_ref(name, load=(VAR_LOAD_PARAMETER, None))

    def load(self, name: str) -> None:
        if self.find_ref(name) is None:
            self._define_ref(name, load=(VAR_LOAD_RESOLVE, name))

    def branch_update(
        self,
        branch_symbols: t.Sequence["Symbols"],
        *,
        exhaustive: bool = False,
        inner_loads: t.Collection[str] = (),
    ) -> None:
        """Merge the symbols of a set of mutually exclusive branches.

        :param branch_symbols: The symbols collected for each branch.
        :param exhaustive: Whether the branches cover every possible path,
            such as an ``{% if %}`` chain ending in ``{% else %}``. If they
            do, a name that every branch assigns before reading does not
            need to be resolved from the outer scope.
        :param inner_loads: Names read inside nested scopes, which may be
            read before a branch assigns them.
        """
        stores: set[str] = set()

        for branch in branch_symbols:
            stores.update(branch.stores)

        stores.difference_update(self.stores)

        if exhaustive:
            # A name assigned by every branch is always defined afterwards,
            # so it does not need to be resolved from the outer scope.
            stores -= {
                name
                for name in stores
                if name not in inner_loads
                and all(_defines_local(branch, name) for branch in branch_symbols)
            }

        for sym in branch_symbols:
            self.refs.update(sym.refs)
            self.loads.update(sym.loads)
            self.stores.update(sym.stores)

        for name in stores:
            target = self.find_ref(name)
            assert target is not None, "should not happen"

            if self.parent is not None:
                outer_target = self.parent.find_ref(name)
                if outer_target is not None:
                    self.loads[target] = (VAR_LOAD_ALIAS, outer_target)
                    continue
            self.loads[target] = (VAR_LOAD_RESOLVE, name)

    def dump_stores(self) -> dict[str, str]:
        rv: dict[str, str] = {}
        node: Symbols | None = self

        while node is not None:
            for name in sorted(node.stores):
                if name not in rv:
                    rv[name] = self.find_ref(name)  # type: ignore

            node = node.parent

        return rv

    def dump_param_targets(self) -> set[str]:
        rv = set()
        node: Symbols | None = self

        while node is not None:
            for target, (instr, _) in self.loads.items():
                if instr == VAR_LOAD_PARAMETER:
                    rv.add(target)

            node = node.parent

        return rv


class RootVisitor(NodeVisitor):
    def __init__(self, symbols: "Symbols") -> None:
        self.sym_visitor = FrameSymbolVisitor(symbols)

    def _simple_visit(self, node: nodes.Node, **kwargs: t.Any) -> None:
        for child in node.iter_child_nodes():
            self.sym_visitor.visit(child)

    visit_Template = _simple_visit
    visit_Block = _simple_visit
    visit_Macro = _simple_visit
    visit_FilterBlock = _simple_visit
    visit_Scope = _simple_visit
    visit_If = _simple_visit
    visit_ScopedEvalContextModifier = _simple_visit

    def visit_AssignBlock(self, node: nodes.AssignBlock, **kwargs: t.Any) -> None:
        for child in node.body:
            self.sym_visitor.visit(child)

    def visit_CallBlock(self, node: nodes.CallBlock, **kwargs: t.Any) -> None:
        for child in node.iter_child_nodes(exclude=("call",)):
            self.sym_visitor.visit(child)

    def visit_OverlayScope(self, node: nodes.OverlayScope, **kwargs: t.Any) -> None:
        for child in node.body:
            self.sym_visitor.visit(child)

    def visit_For(
        self, node: nodes.For, for_branch: str = "body", **kwargs: t.Any
    ) -> None:
        if for_branch == "body":
            self.sym_visitor.visit(node.target, store_as_param=True)
            branch = node.body
        elif for_branch == "else":
            branch = node.else_
        elif for_branch == "test":
            self.sym_visitor.visit(node.target, store_as_param=True)
            if node.test is not None:
                self.sym_visitor.visit(node.test)
            return
        else:
            raise RuntimeError("Unknown for branch")

        if branch:
            for item in branch:
                self.sym_visitor.visit(item)

    def visit_With(self, node: nodes.With, **kwargs: t.Any) -> None:
        for target in node.targets:
            self.sym_visitor.visit(target)
        for child in node.body:
            self.sym_visitor.visit(child)

    def generic_visit(self, node: nodes.Node, *args: t.Any, **kwargs: t.Any) -> None:
        raise NotImplementedError(f"Cannot find symbols for {type(node).__name__!r}")


class FrameSymbolVisitor(NodeVisitor):
    """A visitor for `Frame.inspect`."""

    def __init__(self, symbols: "Symbols") -> None:
        self.symbols = symbols
        # Scopes this visitor stops at, in the order they appear. They are
        # only scanned if `visit_If` asks for the names they read, so a
        # template without an if/else chain does no extra work.
        self._inner_scopes: list[nodes.Node] = []
        self._inner_loads: set[str] = set()

    def _track_inner_loads(
        self, *groups: "nodes.Node | t.Iterable[nodes.Node] | None"
    ) -> None:
        """Remember a scope this visitor does not enter.

        Every field the compiler visits with a child frame must be passed
        here, even if the field is also visited into this frame's symbols.
        A read in such a field is not part of this frame's symbols, but it
        still reads this frame's variables, so :meth:`visit_If` needs to
        know about it.
        """
        for group in groups:
            if group is None:
                continue

            if isinstance(group, nodes.Node):
                self._inner_scopes.append(group)
            else:
                self._inner_scopes.extend(group)

    def _get_inner_loads(self) -> set[str]:
        """The names read so far inside scopes this visitor does not enter."""
        while self._inner_scopes:
            node = self._inner_scopes.pop()

            # ``find_all`` only yields descendants, so check the node itself
            # as well. A field such as ``For.test`` may be a bare ``Name``.
            for child in itertools.chain((node,), node.find_all(nodes.Node)):
                if isinstance(child, nodes.NSRef):
                    self._inner_loads.add(child.name)
                elif isinstance(child, nodes.Name) and child.ctx == "load":
                    self._inner_loads.add(child.name)

        return self._inner_loads

    def visit_Name(
        self, node: nodes.Name, store_as_param: bool = False, **kwargs: t.Any
    ) -> None:
        """All assignments to names go through this function."""
        if store_as_param or node.ctx == "param":
            self.symbols.declare_parameter(node.name)
        elif node.ctx == "store":
            self.symbols.store(node.name)
        elif node.ctx == "load":
            self.symbols.load(node.name)

    def visit_NSRef(self, node: nodes.NSRef, **kwargs: t.Any) -> None:
        self.symbols.load(node.name)

    def visit_If(self, node: nodes.If, **kwargs: t.Any) -> None:
        self.visit(node.test, **kwargs)
        original_symbols = self.symbols

        def inner_visit(nodes: t.Iterable[nodes.Node]) -> "Symbols":
            self.symbols = rv = original_symbols.copy()

            for subnode in nodes:
                self.visit(subnode, **kwargs)

            self.symbols = original_symbols
            return rv

        # The parser produces a flat chain: every `elif` is an `If` node in
        # `elif_` with an empty `elif_`/`else_`, and the final `else` body
        # is stored on the outermost node. Track each body as its own
        # branch so that a name assigned by all of them can be recognised.
        branch_symbols = [inner_visit(node.body)]
        exhaustive = bool(node.else_)

        for elif_node in node.elif_:
            if elif_node.elif_ or elif_node.else_:
                # Not a shape the parser produces. Visit it as a whole and
                # make no assumptions about which paths assign a name.
                branch_symbols.append(inner_visit([elif_node]))
                exhaustive = False
            else:
                branch_symbols.append(inner_visit([elif_node.test, *elif_node.body]))

        branch_symbols.append(inner_visit(node.else_ or ()))
        # A name read inside a nested scope, either before this statement or
        # within a branch, may be read before the branch assigns it, so the
        # outer value is still needed.
        self.symbols.branch_update(
            branch_symbols,
            exhaustive=exhaustive,
            inner_loads=self._get_inner_loads() if exhaustive else (),
        )

    def visit_Macro(self, node: nodes.Macro, **kwargs: t.Any) -> None:
        self.symbols.store(node.name)
        self._track_inner_loads(node.body, node.defaults)

    def visit_Import(self, node: nodes.Import, **kwargs: t.Any) -> None:
        self.generic_visit(node, **kwargs)
        self.symbols.store(node.target)

    def visit_FromImport(self, node: nodes.FromImport, **kwargs: t.Any) -> None:
        self.generic_visit(node, **kwargs)

        for name in node.names:
            if isinstance(name, tuple):
                self.symbols.store(name[1])
            else:
                self.symbols.store(name)

    def visit_Assign(self, node: nodes.Assign, **kwargs: t.Any) -> None:
        """Visit assignments in the correct order."""
        self.visit(node.node, **kwargs)
        self.visit(node.target, **kwargs)

    def visit_For(self, node: nodes.For, **kwargs: t.Any) -> None:
        """Visiting stops at for blocks.  However the block sequence
        is visited as part of the outer scope.
        """
        self.visit(node.iter, **kwargs)
        self._track_inner_loads(node.body, node.else_, node.test)

    def visit_CallBlock(self, node: nodes.CallBlock, **kwargs: t.Any) -> None:
        self.visit(node.call, **kwargs)
        self._track_inner_loads(node.body, node.defaults)

    def visit_FilterBlock(self, node: nodes.FilterBlock, **kwargs: t.Any) -> None:
        self.visit(node.filter, **kwargs)
        # `filter` is visited above so that it resolves in this frame, but
        # the compiler evaluates it in a child frame, so it is tracked too.
        self._track_inner_loads(node.body, node.filter)

    def visit_With(self, node: nodes.With, **kwargs: t.Any) -> None:
        for target in node.values:
            self.visit(target)

        self._track_inner_loads(node.body)

    def visit_AssignBlock(self, node: nodes.AssignBlock, **kwargs: t.Any) -> None:
        """Stop visiting at block assigns."""
        self.visit(node.target, **kwargs)
        self._track_inner_loads(node.body, node.filter)

    def visit_Scope(self, node: nodes.Scope, **kwargs: t.Any) -> None:
        """Stop visiting at scopes."""
        self._track_inner_loads(node.body)

    def visit_Block(self, node: nodes.Block, **kwargs: t.Any) -> None:
        """Stop visiting at blocks."""
        # A block body gets a fresh root frame, so it cannot read this
        # frame's variables. It is tracked anyway rather than relying on
        # that.
        self._track_inner_loads(node.body)

    def visit_OverlayScope(self, node: nodes.OverlayScope, **kwargs: t.Any) -> None:
        """Do not visit into overlay scopes."""
        # `context` is evaluated in this frame but is never visited. The
        # body is isolated, and is tracked only to be safe.
        self._track_inner_loads(node.context, node.body)
