import abc
from collections.abc import Iterable
import dataclasses
import inspect
import random
import typing
from typing import Callable


class Variable:
    """
    A literal or a variable containing data.
    """

    def __init__(self, name, dtype):
        self._name = name
        self._dtype = dtype

    @property
    def dtype(self):
        return self._dtype

    @property
    def name(self):
        return self._name

    def to_str(self) -> str:
        return str(self._name)


class Expression(abc.ABC):
    """
    A program component that returns a Variable.
    """

    @abc.abstractmethod
    def to_str(self) -> str: ...


class Statement(abc.ABC):
    """
    A program component that executes but does not return a value.
    """

    @property
    @abc.abstractmethod
    def in_scope_variables(self, child: "Statement") -> list[Variable]: ...

    @property
    @abc.abstractmethod
    def variables(self) -> list[Variable]: ...

    @property
    @abc.abstractmethod
    def sub_statements(self) -> list["Statement"]: ...

    def add_substatement(self, statement, index=None):
        """
        If index is None, adds the statement at a random location.
        """
        raise ValueError("Cannot add sub-statement to this statement type.")

    def swap_substatement(self, old_statement, new_statement):
        """
        Removes the old statement and places the new one in its place.
        """
        raise ValueError("Cannot swap sub-statements for this statement type.")

    def del_substatement(self, statement):
        raise ValueError("Cannot delete sub-statements for this statement type.")

    @property
    @abc.abstractmethod
    def expressions(self) -> list[Expression]: ...

    # Note: Statements are responsible for ensuring that their string outputs
    # are terminated by a newline.
    @abc.abstractmethod
    def to_str(self, indent_level: int) -> str: ...


def dedupe_variable_list(variables: list[Variable]) -> list[Variable]:
    # We cannot use the usual list(set(variables)) to dedupe because that one
    # will alias unique variables with the same name / other hashable fields.
    is_present = lambda items, item: any(x is item for x in items)
    output_list = []
    for v in variables:
        if not is_present(output_list, v):
            output_list.append(v)
    return output_list


@dataclasses.dataclass
class StatementInfo:
    in_scope_variables: list[Variable]
    parent: Statement


def statement_mutation_info(
    statements: list[Statement],
) -> dict[Statement, StatementInfo]:
    # Does recursive depth-first search on the statement list, outputting
    # information about each statement that is useful for mutation.

    def _recurse_sub_statements(s: Statement) -> dict[Statement, StatementInfo]:
        dict_so_far = {}
        for sub_statement in s.sub_statements:
            info = StatementInfo(in_scope_variables=[], parent=s)
            dict_so_far[sub_statement] = info
            # Note: The dict keys will be in the right order, after Python 3.7,
            # because they are in the insertion order and we insert statements
            # in the order in which they appear in the program (DFS order).
            dict_so_far.update(_recurse_sub_statements(sub_statement))
        return dict_so_far

    # Do this in two steps: First get all of the parents, then get the in-scope
    # variables. While this might be possible in a single recursive pass, 2-pass
    # is much simpler.
    output_info = {}
    for statement in statements:
        output_info[statement] = StatementInfo(in_scope_variables=[], parent=None)
        output_info.update(_recurse_sub_statements(statement))
    # Next get all of the in-scope variables.
    # Start by setting the variables for the top-level nodes, which don't have
    # parents.
    top_level_vars = []
    for statement in statements:
        output_meta = output_info[statement]
        output_meta.in_scope_variables = top_level_vars.copy()
        top_level_vars.extend(statement.variables)
    # Now iterate through all of the statements, adding all variables that are
    # in-scope for the parent.
    for statement in output_info.keys():
        statement_meta = output_info[statement]
        # Loop through the parents, adding the parent's vars at every step.
        all_in_scope_vars = []
        parent_statement = statement_meta.parent
        child_statement = statement
        while parent_statement is not None:
            parent_in_scope_vars = []
            # Add any existing variables that are in scope for the parent as
            # well as all its children.
            parent_in_scope_vars.extend(
                output_info[parent_statement].in_scope_variables
            )
            # Add any variables that are defined within the parent and in-scope
            # for this child.
            parent_in_scope_vars.extend(
                parent_statement.in_scope_variables(child=child_statement)
            )
            all_in_scope_vars.extend(parent_in_scope_vars)
            # Make the parent the new child to traverse up the tree.
            child_statement = parent_statement
            parent_statement = output_info[child_statement].parent

        deduped_vars = dedupe_variable_list(all_in_scope_vars)
        statement_meta.in_scope_variables.extend(deduped_vars)
    return output_info


def add_indentation(string, level=1):
    statements = string.split("\n")
    statements = ["  " * level + s for s in statements]
    return "\n".join(statements)


class CallableExpression(Expression):
    """A function that operates on one or more Variables."""

    def __init__(
        self,
        input_args: list[Variable],
        transform_fn: Callable,
    ):
        self._input_args = input_args
        self._transform_fn = transform_fn

    def to_str(self) -> str:
        func_name = self._transform_fn.__name__
        arg_names = [a.to_str() for a in self._input_args]
        return f"{func_name}({', '.join(arg_names)})"


class PlaceholderStatement(Statement):
    """This is only meant to be used temporarily, while mutating a program."""

    def to_str(self, indent_level=0) -> str:
        s = add_indentation("PLACEHOLDER", indent_level)
        return s + "\n"

    @property
    def in_scope_variables(self, child: "Statement") -> list[Variable]:
        return []

    @property
    def variables(self) -> list[Variable]:
        return []

    @property
    def sub_statements(self) -> list["Statement"]:
        return []

    @property
    def expressions(self) -> list[Expression]:
        return []


class AssignmentStatement(Statement):
    def __init__(
        self,
        left_side: Variable,
        right_side: Expression | Variable,
    ):
        self._left_side = left_side
        self._right_side = right_side

    def to_str(self, indent_level=0) -> str:
        s = f"{self._left_side.to_str()} = {self._right_side.to_str()}"
        s = add_indentation(s, indent_level)
        return s + "\n"

    @property
    def variables(self) -> list[Variable]:
        output = [self._left_side]
        if isinstance(self._right_side, Variable):
            output.append(self._right_side)
        return output

    @property
    def sub_statements(self) -> list[Statement]:
        return []

    @property
    def expressions(self) -> list[Expression]:
        output = []
        if self.isinstance(self._right_side, Expression):
            output.append(self._right_side)
        return output

    def in_scope_variables(self, child: Statement) -> list[Variable]:
        # AssignmentStatements do not have children.
        return []


class ForEachStatement(Statement):
    """A statement that loops over an iterable."""

    def __init__(
        self,
        item_variable: Variable,
        iterable: Variable | Expression,
        statement_list: list[Statement],
    ):
        self._statements = statement_list
        self._iterable = iterable
        self._item_variable = item_variable

    def to_str(self, indent_level=0) -> str:
        output_str = f"for {self._item_variable.to_str()} "
        output_str += f"in {self._iterable.to_str()}:"
        output_str = add_indentation(output_str, level=indent_level)
        output_str += "\n"
        if not self._statements:
            output_str += add_indentation("pass", level=indent_level + 1)
            output_str += "\n"
        for statement in self._statements:
            substatement_str = statement.to_str(indent_level=indent_level + 1)
            output_str += substatement_str
        return output_str

    @property
    def variables(self) -> list[Variable]:
        # This is fine because the loop item remains in scope after the for-loop
        # exits, due to how Python treats blocks and variable scope.
        return [self._item_variable]

    @property
    def sub_statements(self) -> list[Statement]:
        output_statements = []
        for statement in self._statements:
            output_statements.append(statement)
        return output_statements

    @property
    def expressions(self) -> list[Expression]:
        output = []
        if self.isinstance(self._iterable, Expression):
            output.append(self._iterable)
        return output

    def in_scope_variables(self, child: Statement) -> list[Variable]:
        variables = [self._item_variable]
        for statement in self._statements:
            if statement is child:
                break
            variables.extend(statement.variables)
        return dedupe_variable_list(variables)

    def add_substatement(self, statement, index=None):
        if index is None:
            # Then we will add the substatement at a random location.
            index = random.randint(0, len(self._statements))
        self._statements.insert(index, statement)

    def swap_substatement(self, old_statement, new_statement):
        """Removes the old statement and places the new one in its place."""
        for idx, statement in enumerate(self._statements):
            if old_statement is statement:
                self._statements[idx] = new_statement
                return
        raise ValueError("Old statement is not present in the list of statements.")

    def del_substatement(self, statement):
        for idx, candidate in enumerate(self._statements):
            if candidate is statement:
                del self._statements[idx]
                return
        raise ValueError("Statement is not present in the list of statements.")


class IfElseStatement(Statement):
    """A statement that executes if a condition evaluates to true."""

    def __init__(
        self,
        condition: Variable | Expression,
        if_statements: list[Statement],
        else_statements: list[Statement],
    ):
        self._condition = condition
        self._if_statements = if_statements
        self._else_statements = else_statements

    def to_str(self, indent_level=0) -> str:
        output_str = f"if {self._condition.to_str()}:"
        output_str = add_indentation(output_str, level=indent_level)
        output_str += "\n"
        if not self._if_statements:
            output_str += add_indentation("pass", level=indent_level + 1)
            output_str += "\n"
        for statement in self._if_statements:
            substatement_str = statement.to_str(indent_level=indent_level + 1)
            output_str += substatement_str
        if self._else_statements:
            output_str += add_indentation("else:", level=indent_level)
            output_str += "\n"
            for statement in self._else_statements:
                substatement_str = statement.to_str(indent_level=indent_level + 1)
                output_str += substatement_str
        return output_str

    @property
    def variables(self) -> list[Variable]:
        return []

    @property
    def sub_statements(self) -> list[Statement]:
        output_statements = []
        for statement in self._if_statements:
            output_statements.append(statement)
        for statement in self._else_statements:
            output_statements.append(statement)
        return output_statements

    @property
    def expressions(self) -> list[Expression]:
        output = []
        if self.isinstance(self._iterable, Expression):
            output.append(self._iterable)
        return output

    @property
    def iterable(self):
        return self._iterable

    @property
    def condition(self):
        return self._condition

    def in_scope_variables(self, child: Statement) -> list[Variable]:
        # First determine if the statement is in the if branch or the else
        # branch (or neither, in which case we raise a ValueError).
        is_in_if = any([child is s for s in self._if_statements])
        is_in_else = any([child is s for s in self._else_statements])
        if is_in_if and is_in_else:
            raise ValueError(
                "Child statement is in both the if branch and the else branch"
            )
        if not is_in_if and not is_in_else:
            raise ValueError(
                "Child statement is in neither the if branch or the else branch"
            )
        statements = self._if_statements if is_in_if else self._else_statements
        variables = []
        for statement in statements:
            if statement is child:
                break
            variables.extend(statement.variables)
        return dedupe_variable_list(variables)

    def add_substatement(self, statement, index=None):
        # For if/else, index has to be 2D: (block_index, statement_index)
        # where block_index is 0 if adding to the "if" part and 1 if adding to "else"
        if index is None:
            # Then we will add the substatement at a random location in a random block.
            index_0 = random.randint(0, 1)
            statements = self._if_statements if index_0 else self._else_statements
            index_1 = random.randint(0, len(statements))
            index = (index_0, index_1)
        index_0, index_1 = index
        if index_0:
            self._if_statements.insert(index_1, statement)
        else:
            self._else_statements.insert(index_1, statement)

    def del_substatement(self, statement):
        for idx, candidate in self._if_statements:
            if candidate is statement:
                del self._if_statements[idx]
                return
        for idx, candidate in self._else_statements:
            if candidate is statement:
                del self._else_statements[idx]
                return
        raise ValueError("Statement is not present in the list of statements.")

    def swap_substatement(self, old_statement, new_statement):
        """Removes the old statement and places the new one in its place."""
        for idx, statement in enumerate(self._if_statements):
            if old_statement is statement:
                self._if_statements[idx] = new_statement
                return
        for idx, statement in enumerate(self._else_statements):
            if old_statement is statement:
                self._else_statements[idx] = new_statement
                return
        raise ValueError("Old statement is not present in the list of statements.")


###############################################################################
# Functions to sample and mutate a valid expression / statement.
###############################################################################


def sample_callable_expression(
    in_scope_variables: list[Variable],
    callables_library: list[Callable],
    return_type,
) -> CallableExpression | None:
    eligible_callables = []
    for c in callables_library:
        sig = inspect.signature(c)
        ret_type_ok = sig.return_annotation == return_type
        all_args_exist = True
        for param_name, param_value in sig.parameters.items():
            param_type = param_value.annotation
            input_exists = any(v.dtype == param_type for v in in_scope_variables)
            all_args_exist &= input_exists
        # If both ret_type_ok and all_args_exist then it is possible
        # to call this expression without an undefined reference or bad
        # return type.
        if all_args_exist and ret_type_ok:
            eligible_callables.append(c)
    if not eligible_callables:
        return None
    # All of these callables will return the correct type and can be
    # called with the correct arguments. Now we just pick one!
    call_fn = random.choice(eligible_callables)
    arguments = []
    sig = inspect.signature(call_fn)
    for param_name, param_value in sig.parameters.items():
        param_type = param_value.annotation
        eligible_args = [v for v in in_scope_variables if v.dtype == param_type]
        arguments.append(random.choice(eligible_args))
    return CallableExpression(arguments, call_fn)


def sample_foreach_statement(
    in_scope_variables: list[Variable],
    callables_library: list[Callable],
    item_variable_name: str,
) -> Statement | None:
    eligible_iterables = []
    for v in in_scope_variables:
        if typing.get_origin(v.dtype) == list:
            eligible_iterables.append(v)
    if not eligible_iterables:
        return None
    iterable_variable = random.choice(eligible_iterables)
    arg_type = typing.get_args(iterable_variable.dtype)
    if len(arg_type) != 1:
        raise ValueError("All list types must be fully-specified, with one item type.")
    item_variable = Variable(name=item_variable_name, dtype=arg_type)
    return ForEachStatement(
        item_variable,
        iterable_variable,
        [],
    )


def sample_initialization_statement(
    in_scope_variables: list[Variable],
    callables_library: list[Callable],
    new_variable_name: str,
) -> Statement | None:
    # Since there is no constraint on the return variable type (we are making
    # a new variable), we can call any function we want.
    eligible_callables = []
    for c in callables_library:
        sig = inspect.signature(c)
        all_args_exist = True
        for param_name, param_value in sig.parameters.items():
            param_type = param_value.annotation
            input_exists = any(v.dtype == param_type for v in in_scope_variables)
            all_args_exist &= input_exists
        # If both all_args_exist then it is possible to call this
        # expression without an undefined reference.
        if all_args_exist:
            eligible_callables.append(c)
    if not eligible_callables:
        return None
    # Pick a callable and make it into an expression.
    call_fn = random.choice(eligible_callables)
    arguments = []
    sig = inspect.signature(call_fn)
    for param_name, param_value in sig.parameters.items():
        param_type = param_value.annotation
        eligible_args = [v for v in in_scope_variables if v.dtype == param_type]
        arguments.append(random.choice(eligible_args))
    expr = CallableExpression(arguments, call_fn)
    new_var = Variable(name=new_variable_name, dtype=sig.return_annotation)
    # We don't allow for the RHS of the AssignmentStatement to be a Variable because
    # of how Python handles copy-by-reference. It's better to make this an explicit
    # typed make_copy() callable expression.
    return AssignmentStatement(
        new_var,
        expr,
    )


def sample_reassignment_statement(
    in_scope_variables: list[Variable],
    callables_library: list[Callable],
) -> Statement | None:
    # We pick one of the in-scope variables and we re-define it to be the result
    # of an AssignmentStatement.
    if not in_scope_variables:
        return None
    var_to_reassign = random.choice(in_scope_variables)
    expr = sample_callable_expression(
        in_scope_variables, callables_library, var_to_reassign.dtype
    )
    if expr is None:
        return None
    new_var = Variable(name=var_to_reassign.name, dtype=var_to_reassign.dtype)
    return AssignmentStatement(
        new_var,
        expr,
    )


def sample_if_else_statement(
    in_scope_variables: list[Variable],
    callables_library: list[Callable],
) -> Statement | None:

    condition_expr = sample_callable_expression(
        in_scope_variables,
        callables_library,
        bool,
    )
    if condition_expr is None:
        return None
    return IfElseStatement(condition_expr, [], [])


def sample_new_statement(
    in_scope_variables: list[Variable],
    callables_library: list[Callable],
    new_variable_id: int = 0,  # Increment each call, to avoid having duplicate variable names.
    max_attempts: int = 20,  # Max number of attempts to create a new statement.
    action_weights: dict[str, float] | None = None,
) -> Statement | None:
    eligible_statement_types = ["INIT", "REASSIGN", "IF_ELSE", "FOR_EACH"]
    if action_weights:
        eligible_statement_weights = [
            action_weights[t] for t in eligible_statement_types
        ]
    else:
        eligible_statement_weights = [1] * len(eligible_statement_types)
    output_statement = None
    for _ in range(max_attempts):
        # Select a type of statement to add.
        selection = random.choices(
            eligible_statement_types,
            weights=eligible_statement_weights,
            k=1,
        )[0]
        match selection:
            case "INIT":
                output_statement = sample_initialization_statement(
                    in_scope_variables,
                    callables_library,
                    f"x_{new_variable_id}",
                )
            case "REASSIGN":
                output_statement = sample_reassignment_statement(
                    in_scope_variables,
                    callables_library,
                )
            case "FOR_EACH":
                output_statement = sample_foreach_statement(
                    in_scope_variables,
                    callables_library,
                    f"item_{new_variable_id}",
                )
            case "IF_ELSE":
                output_statement = sample_if_else_statement(
                    in_scope_variables,
                    callables_library,
                )
            case _:
                raise ValueError("Unsupported statement type for mutation: ", selection)
        if output_statement is not None:
            return output_statement
    return output_statement


def mutate_assignment_statement(
    in_scope_variables,
    callables_library,
    statement: AssignmentStatement,
) -> AssignmentStatement | None:
    left_hand_side = statement.variables[0]
    expr = sample_callable_expression(
        in_scope_variables,
        callables_library,
        left_hand_side.dtype,
    )
    if expr is None:
        return None
    return AssignmentStatement(
        left_hand_side,
        expr,
    )


def mutate_if_else_statement(
    in_scope_variables,
    callables_library,
    statement: IfElseStatement,
) -> IfElseStatement | None:
    expr = sample_callable_expression(
        in_scope_variables,
        callables_library,
        bool,
    )
    if expr is None:
        return None
    # TODO: Fix protected variable access here.
    return IfElseStatement(
        expr,
        statement._if_statements.copy(),
        statement._else_statements.copy(),
    )


def mutate_for_each_statement(
    in_scope_variables,
    callables_library,
    statement: ForEachStatement,
) -> ForEachStatement | None:
    item_type = statement._item_variable.dtype
    eligible_iterables = []
    for v in in_scope_variables:
        if v.dtype == list[item_type]:
            eligible_iterables.append(v)
    if not eligible_iterables:
        return None
    new_iterable = random.choice(eligible_iterables)
    return ForEachStatement(
        statement._item_variable,
        new_iterable,
        statement._statements.copy(),
    )


def mutate_statement(
    in_scope_variables: list[Variable],
    callables_library: list[Callable],
    statement: Statement,
) -> Statement | None:
    if isinstance(statement, AssignmentStatement):
        return mutate_assignment_statement(
            in_scope_variables, callables_library, statement
        )
    elif isinstance(statement, ForEachStatement):
        return mutate_for_each_statement(
            in_scope_variables, callables_library, statement
        )
    elif isinstance(statement, IfElseStatement):
        return mutate_if_else_statement(
            in_scope_variables, callables_library, statement
        )
    else:
        raise ValueError("Cannot mutate this statement type.")


###############################################################################
# Pruning algorithm definition.
###############################################################################


class PruningAlgorithm:
    def __init__(self, statements=[]):
        self._statements = statements
        self._input_variable = Variable("input_candidates", list[Node])
        self._output_variable = self._input_variable
        self._output_type = list[Node]
        self._iter_number = 0

    def update_return_var(self, var: Variable):
        self._output_variable = var

    @property
    def sub_statements(self):
        return self._statements

    @sub_statements.setter
    def sub_statements(self, value):
        self._statements = value

    def add_substatement(self, statement, index=None):
        if index is None:
            # Then we will add the substatement at a random location.
            index = random.randint(0, len(self._statements))
        self._statements.insert(index, statement)

    def swap_substatement(self, old_statement, new_statement):
        """Removes the old statement and places the new one in its place."""
        for idx, statement in enumerate(self._statements):
            if old_statement is statement:
                self._statements[idx] = new_statement
                return
        raise ValueError("Old statement is not present in the list of statements.")

    def del_substatement(self, statement):
        for idx, candidate in enumerate(self._statements):
            if candidate is statement:
                del self._statements[idx]
                return
        raise ValueError("Statement is not present in the list of statements.")

    def in_scope_variables(self, child: Statement) -> list[Variable]:
        variables = [self._input_variable]
        for statement in self._statements:
            if statement is child:
                break
            variables.extend(statement.variables)
        return dedupe_variable_list(variables)

    @property
    def input_var(self):
        return self._input_variable

    def to_str(self):
        output_str = f"def prune({self._input_variable.to_str()}):\n"
        for statement in self._statements:
            output_str += statement.to_str(indent_level=1)
        output_str += f"  return take_first_M({self._output_variable.to_str()})\n"
        return output_str

    def add_new_statement(
        self,
        callables_library: list[Callable],
        action_weights: dict[str, float] | None = None,
        max_attempts=20,
    ):
        # 1. Pick a location to add the new statement.
        # This is slightly tricky, because we can add the new statement at
        # various levels - either at the top level, or within various loops or
        # conditional statements. The LION paper picks a random index where the
        # statement will get inserted, but this is hard to do when we have
        # nested logic. Instead, we first list the eligible parent statements,
        # pick one of those, and then pick a random location in our selection.
        statement_info = statement_mutation_info(self._statements)
        all_statements = list(statement_info.keys())
        eligible_parents = [self]
        prob_weights = [len(self.sub_statements) + 1]
        for statement in all_statements:
            # We weight the selection by the line count to be sure that we do
            # a uniform random selection of locations to add a statement.
            is_eligible = lambda s: not isinstance(s, AssignmentStatement)
            if is_eligible(statement):
                eligible_parents.append(statement)
                prob_weights.append(1 + len(statement.sub_statements))

        for _ in range(max_attempts):
            parent_choice = random.choices(eligible_parents, weights=prob_weights, k=1)[
                0
            ]
            # Now select a random location inside the selected parent.
            placeholder = PlaceholderStatement()
            parent_choice.add_substatement(placeholder, index=None)
            # 2. Find all of the variables that are in-scope at the insertion point,
            # to avoid writing a program that throws an undefined reference.
            in_scope_variables = [self._input_variable]
            if parent_choice is not self:
                in_scope_variables.extend(
                    statement_info[parent_choice].in_scope_variables
                )
            in_scope_variables.extend(parent_choice.in_scope_variables(placeholder))
            # 3. Pick a type of new statement to add.
            new_statement = sample_new_statement(
                in_scope_variables,
                callables_library,
                action_weights=action_weights,
                new_variable_id=self._iter_number,
            )
            if new_statement is None:
                parent_choice.del_substatement(placeholder)
                continue
            # 4. Add the new statement at the specified location.
            parent_choice.swap_substatement(placeholder, new_statement)
            self._iter_number += 1
            return

    def mutate_existing_statement(
        self, callables_library: list[Callable], max_attempts=20
    ):
        statement_info = statement_mutation_info(self._statements)
        all_statements = list(statement_info.keys())
        for _ in range(max_attempts):
            statement_to_mutate = random.choice(all_statements)
            info = statement_info[statement_to_mutate]
            try:
                mutated_statement = mutate_statement(
                    info.in_scope_variables,
                    callables_library,
                    statement_to_mutate,
                )
            except:
                continue
            if mutate_statement is not None:
                if info.parent is not None:
                    info.parent.swap_substatement(
                        statement_to_mutate, mutated_statement
                    )
                else:
                    self.swap_substatement(statement_to_mutate, mutated_statement)
                return

    # def delete_existing_statement(self, callables_library: list[Callable], max_attempts=20):

    def mutate(
        self,
        callables_library: list[Callable],
        add_action_weights: dict[str, float] | None = None,
        mutate_action_weights: dict[str, float] | None = None,
        max_attempts: int = 20,
    ):
        actions = ["ADD_NEW", "MUTATE"]
        if mutate_action_weights:
            weights = [mutate_action_weights[a] for a in actions]
        else:
            weights = [1] * len(actions)
        action = random.choices(actions, weights=weights, k=1)[0]
        match action:
            case "ADD_NEW":
                self.add_new_statement(
                    callables_library,
                    max_attempts=max_attempts,
                    action_weights=add_action_weights,
                )
            case "MUTATE":
                self.mutate_existing_statement(
                    callables_library,
                    max_attempts=max_attempts,
                )
            case _:
                raise ValueError("Unknown action.")
        # If the return variable is no longer in-scope (maybe no longer present)
        # then update the return variable.
        placeholder = PlaceholderStatement()
        self._statements.append(placeholder)
        in_scope_at_end = self.in_scope_variables(placeholder)
        del self._statements[-1]
        output_variable_exists = False
        for v in in_scope_at_end:
            output_variable_exists |= self._output_variable is v
        if not output_variable_exists:
            # Re-assign output_variable to some other variable that does exist.
            eligible_vars = []
            for v in in_scope_at_end:
                if v.dtype == self._output_type:
                    if v is not self._input_variable:
                        eligible_vars.append(v)
            if not eligible_vars:
                raise ValueError(
                    "All possible candidates for the return value " "have been deleted."
                )
            self._output_variable = random.choice(eligible_vars)
        return


###############################################################################
# Define the node pruning algorithm and eligible mutations.
###############################################################################


class Node:
    """Datatype for (node ID, distance) that supports distance computations."""
    def __init__(self, node_id: int, distance: float):
        self.node_id = node_id
        self.distance = distance
    
    def __eq__(self, other):
        return self.node_id == other.node_id


# None of these are implemented, this is just to demonstrate.
def greater_or_equal(x: float, y: float) -> bool:
    return False


def sort_by_distance_asc(x: list[Node]) -> list[Node]:
    return x


def empty_node_list() -> list[Node]:
    return []


def sort_by_distance_desc(x: list[Node]) -> list[Node]:
    return x


def argmin_distance(x: list[Node]) -> Node:
    return x[0]


def argmax_distance(x: list[Node]) -> Node:
    return x[0]


def min_distance(x: list[Node], y: Node) -> float:
    return 0.0


def max_distance(x: list[Node], y: Node) -> float:
    return 0.0


def median_distance(x: list[Node], y: Node) -> float:
    return 0.0


def mean_distance(x: list[Node], y: Node) -> float:
    return 0.0


def append_node(x: list[Node], y: Node) -> list[Node]:
    x.append(y)
    return x


def take_first_M(x: list[Node]) -> list[Node]:
    return x


def distance_to_query(x: Node) -> float:
    return 0.0


ALL_CALLABLES = [
    greater_or_equal,
    sort_by_distance_asc,
    empty_node_list,
    sort_by_distance_desc,
    argmin_distance,
    argmax_distance,
    min_distance,
    max_distance,
    median_distance,
    mean_distance,
    append_node,
    take_first_M,
    distance_to_query,
]


def assign(variable, rhs):
    return AssignmentStatement(variable, rhs)


def call(callable_fn, *var_list):
    return CallableExpression(var_list, callable_fn)


def build_beam_search():
    algorithm = PruningAlgorithm()
    statements = []
    # sorted_candidates = sort_by_distance_asc(input_candidates)
    sorted_candidates = Variable("sorted_candidates", list[Node])
    saved_candidates = Variable("saved_candidates", list[Node])
    op = assign(saved_candidates, call(empty_node_list))
    statements.append(op)
    op = assign(sorted_candidates, call(sort_by_distance_asc, algorithm.input_var))
    statements.append(op)
    # for candidate in sorted_candidates:
    candidate = Variable("candidate", Node)  # Iterator item.
    baseline_distance = Variable("baseline_distance", float)  # Var declaration.
    closest_saved_candidate_dist = Variable(
        "closest_saved_candidate_dist", float
    )  # Var declaration.
    loop = ForEachStatement(
        candidate,  # in
        sorted_candidates,  # do
        [
            # baseline_distance = distance_to_query(candidate)
            assign(baseline_distance, call(distance_to_query, candidate)),
            # closest_saved_candidate_dist = min_distance(saved_candidates, candidate)
            assign(
                closest_saved_candidate_dist,
                call(min_distance, saved_candidates, candidate),
            ),
            # if closest_saved_candidate_distance >= baseline_distance:
            IfElseStatement(
                call(greater_or_equal, closest_saved_candidate_dist, baseline_distance),
                if_statements=[
                    # append_node(candidate, saved_candidates)
                    assign(
                        saved_candidates, call(append_node, candidate, saved_candidates)
                    ),
                ],
                else_statements=[],
            ),
        ],
    )
    statements.append(loop)
    algorithm.update_return_var(saved_candidates)
    algorithm.sub_statements = statements
    return algorithm


def build_select_neighbors_heuristic():
    """
    Build the base HNSW select neighbors heuristic.
    """
    pass


algo = build_beam_search()
print("Original beam search algorithm:")
print(algo.to_str())


# action_weights = {"INIT": 90, "REASSIGN": 10, "IF_ELSE": 0, "FOR_EACH": 0}
# for i in range(20):
#     algo.add_new_statement(
#         callables_library=ALL_CALLABLES,
#         action_weights=action_weights,
#     )

# # '''
# for i in range(20):
#     print(f"Iteration {i}")
#     algo.mutate(
#         callables_library=ALL_CALLABLES,
#         add_action_weights=action_weights,
#         mutate_action_weights={"ADD_NEW": 30, "MUTATE": 70},
#     )
#     print(algo.to_str())
# '''

"""
statement_info = statement_mutation_info(algo.statements)
for key, info in statement_info.items():
    print(key.to_str().split("\n")[0], f"({key})")
    print("\t", [v._name for v in info.in_scope_variables])
    new_state = sample_foreach_statement(info.in_scope_variables, ALL_CALLABLES, "item_0")
    if new_state is not None:
        print("\tPossible new FOR-LOOP:")
        print(new_state.to_str(indent_level=3))
        print("")
    # expr = sample_callable_expression(info.in_scope_variables, ALL_CALLABLES, list[Node])
    # if expr is not None:
    #     print("\tPossible new EXPR:")
    #     print("\t\t", expr.to_str())
    #     print("")
    new_state = sample_initialization_statement(info.in_scope_variables, ALL_CALLABLES, "new_var")
    if new_state is not None:
        print("\tPossible new INIT:")
        print(new_state.to_str(indent_level=3))
        print("")
    new_state = sample_reassignment_statement(info.in_scope_variables, ALL_CALLABLES)
    if new_state is not None:
        print("\tPossible new REASSIGN:")
        print(new_state.to_str(indent_level=3))
        print("")
# """
