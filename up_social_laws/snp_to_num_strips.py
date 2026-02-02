import itertools
from collections import defaultdict
from fractions import Fraction

import pandas as pd
from unified_planning.model import Problem, InstantaneousAction, OperatorKind, Fluent, Object
from unified_planning.model.multi_agent import Agent, MultiAgentProblem
from unified_planning.shortcuts import Plus, Equals, Minus, Times, LT, LE, GT, GE, Dot, RealType

ONE = ('1', ())

def operator_to_symbol(operator):
    return {OperatorKind.LT:'<', OperatorKind.LE:'<=', OperatorKind.EQUALS: '='}[operator]

def remove_counter_suffix(col_name):
    if col_name.split('#')[-1].isdigit():
        return col_name.rsplit('#', 1)[0]
    return col_name

def is_comparison(node):
    return node.node_type in [OperatorKind.LE, OperatorKind.EQUALS, OperatorKind.LT]

def constant_value(const):
    v = const.type.lower_bound
    if v == const.type.upper_bound:
        return v
    raise Exception("Couldn\'t parse constant")

def is_constant(node):
    return node.node_type in [OperatorKind.REAL_CONSTANT, OperatorKind.INT_CONSTANT]

def simple_fluent(node):
    return node.node_type in [OperatorKind.FLUENT_EXP]

def normalize_dataframe(df, dont_change_columns=None, ignore_as_divisor_columns=None,  add_coeffs=True, operator_column=None):
    if dont_change_columns is None:
        dont_change_columns = []

    if ignore_as_divisor_columns is None:
        ignore_as_divisor_columns = []

    df_frac = df.copy()
    coeffs = []

    # Identify numeric columns only, excluding ignored ones
    numeric_cols = []
    for col in df_frac.columns:
        col_objects = list(df_frac[col])
        if all([isinstance(obj, int) or isinstance(obj, float) for obj in col_objects]):
            numeric_cols.append(col)

    # Convert all numeric entries in relevant columns to Fraction
    for col in numeric_cols:
        df_frac[col] = df_frac[col].apply(lambda x: Fraction(str(x)))

    # Normalize each row
    for i, row in df_frac.iterrows():
        divisor = None
        # find leftmost non-zero numeric value (excluding ignored)
        for col in numeric_cols:
            if row[col] != 0 and col not in ignore_as_divisor_columns:
                divisor = row[col]
                break
        if divisor is not None:
            for col in numeric_cols:
                if col not in dont_change_columns:
                    df_frac.at[i, col] = row[col] / divisor
        coeffs.append(divisor)
        if operator_column is not None and divisor < 0:
            df_frac.at[i, operator_column] = {'<=': '>=', '>=': '<=', '<': '>', '>': '<', '=': '='}[row[operator_column]]
    if add_coeffs:
        df_frac['coeff'] = coeffs

    return df_frac

def get_operator_as_function(op):
    return {OperatorKind.PLUS: Plus,
            OperatorKind.EQUALS: Equals,
            OperatorKind.MINUS: Minus,
            OperatorKind.TIMES: Times,
            OperatorKind.LT: LT,
            OperatorKind.LE: LE,
            '>':GT,
            '>=':GE,
            '<': LT,
            '<=': LE,
            '=':Equals
            }[op]

def get_fluent_from_simple_fluent(fluent):
    return str(fluent).split('(')[0]

def is_operator(node):
    return node.node_type in [OperatorKind.PLUS, OperatorKind.MINUS, OperatorKind.TIMES]

def parse_linear_expression(lin):
    stack = [(lin, 1)]  # Each item is (sub_expr, multiplier)
    coeffs = {ONE: 0}
    while stack:
        node, coeff = stack.pop()

        if is_constant(node):
            coeffs[ONE] += coeff * constant_value(node)

        elif simple_fluent(node):
            fluent = AtomicFluent.from_fnode(node)
            fluent_repr = fluent.to_tuple()
            coeffs[fluent_repr] = coeffs.get(fluent_repr, 0) + coeff

        elif is_operator(node):

            a, b = node.args
            if node.node_type in [OperatorKind.PLUS, OperatorKind.MINUS]:
                coeff_b = coeff
                if node.node_type == OperatorKind.MINUS:
                    coeff_b *= -1
                stack.append((a, coeff))
                stack.append((b, coeff_b))

            elif node.node_type == OperatorKind.TIMES:
                if is_constant(a):
                    stack.append((b, coeff * constant_value(a)))
                elif is_constant(b):
                    stack.append((a, coeff * constant_value(b)))
                else:
                    raise ValueError("Non-linear multiply detected: both operands are non-constants")
            else:
                raise ValueError(f"Unknown operator")
        else:
            raise ValueError(f"Invalid expression node: {node}")

    return coeffs

class AtomicFluent:

    @classmethod

    def from_fnode(cls, node):
        af = AtomicFluent(str(node).split('(')[0].split('.')[-1], tuple(str(a) for a in node.args))
        af.args = node.args
        af.type = node.type
        return af

    def __init__(self, fluent_name, arg_names):
        self._fluent_name = fluent_name
        self._args_names = arg_names
        self.args = []
        self._type = None

    def to_tuple(self):
        return self._fluent_name, self._args_names

    def set_new_args(self, parameters, objects):
        self.args = []
        parameters = {p.name: p for p in parameters}
        objects = {o.name: o for o in objects}
        for arg_name in self._args_names:
            if arg_name in parameters:
                self.args.append(parameters[arg_name])
            elif arg_name in objects:
                self.args.append(objects[arg_name])
            else:
                raise Exception(f"Failed to parse argument: {arg_name}")

class NumStripsPreconditionData:
    def __init__(self, fluent_expr, agent_name=None, is_waitfor=False):
        self.is_waitfor = is_waitfor
        self.fluent_expr = fluent_expr
        self.args = []
        self.fluent = None
        self._agent_name = agent_name


class SimpleBoolPreconditionData(NumStripsPreconditionData):
    def __init__(self, fluent_expr, agent_name=None,is_waitfor=False):
        assert (fluent_expr.node_type == OperatorKind.FLUENT_EXP and str(fluent_expr.type) == "bool")
        NumStripsPreconditionData.__init__(self, fluent_expr, agent_name=agent_name, is_waitfor=is_waitfor)
        self.target_fluent = AtomicFluent.from_fnode(fluent_expr)


class LinearPreconditionData(NumStripsPreconditionData):

    def __init__(self, fluent_expr, agent_name=None,is_waitfor=False):
        NumStripsPreconditionData.__init__(self, fluent_expr, agent_name=agent_name, is_waitfor=is_waitfor)
        if not is_comparison(fluent_expr):
            raise Exception("Not a comparison precondition!")
        lin_node1, lin_node2 = fluent_expr.args
        lin1 = parse_linear_expression(lin_node1)
        lin2 = parse_linear_expression(lin_node2)
        prec_simple_fluents = set(lin1.keys()) | set(lin2.keys())
        self.operator = fluent_expr.node_type
        self.sub_fluents_coeffs = {}
        for p in prec_simple_fluents:
            if p != ONE:
                self.sub_fluents_coeffs[p] = lin1.get(p, 0) - lin2.get(p, 0)
        self.value = lin2.get(ONE, 0) - lin1.get(ONE, 0)


class NumStripsEffectData:
    def __init__(self, fluent_expr):
        self.fluent_expr = fluent_expr

class SimpleBoolEffectData(NumStripsEffectData):

    @classmethod
    def from_init_val(cls, fnode, val):
        assert val.node_type == OperatorKind.BOOL_CONSTANT
        data = SimpleBoolEffectData(None)
        data.target_fluent = AtomicFluent.from_fnode(fnode)
        data.value = str(val)=='true'
        return data

    def __init__(self, fluent_expr):
        super().__init__(fluent_expr)
        self.target_fluent = None
        self.value = None
        if fluent_expr is not None:
            value = fluent_expr.value
            assert value.node_type == OperatorKind.BOOL_CONSTANT
            self.target_fluent = AtomicFluent.from_fnode(fluent_expr.fluent)
            self.value = str(value) == 'true'


class LinearEffectData(NumStripsEffectData):

    @classmethod
    def from_init_val(cls, fnode, val):
        data = LinearEffectData(None)
        data.target_fluent = AtomicFluent.from_fnode(fnode)
        assert is_constant(val)
        data.change = val
        return data

    def __init__(self, fluent_expr):
        NumStripsEffectData.__init__(self, fluent_expr)
        self.target_fluent = None
        self.change = 0
        if fluent_expr is not None:

            target_var, expr_node = fluent_expr.fluent, fluent_expr.value
            self.target_fluent = AtomicFluent.from_fnode(target_var)

            try:
                lin_expr = parse_linear_expression(expr_node)
            except Exception:
                raise Exception(f'Effect {target_var}={expr_node} is not convertible to numeric STRIPS.')
            target_fluent_repr = self.target_fluent.to_tuple()
            assert target_fluent_repr in lin_expr
            assert set(lin_expr.keys()) == {ONE, target_fluent_repr} and lin_expr[target_fluent_repr] == 1
            self.change = lin_expr[ONE]


class NumericStripsActionData:

    def __init__(self, action=None):
        self.parameters = {}
        self.name = None
        self._source_action = action
        if action is not None:
            self.parameters = {p.name: p.type for p in action.parameters}
            self.name = action.name
        self.boolean_preconditions = []
        self.linear_preconditions = []
        self.boolean_effects = []
        self.linear_effects = []

    def parse_preconditions(self):
        for prec in self._source_action.preconditions:
            for prec_parser, preconditions_list in [(SimpleBoolPreconditionData, self.boolean_preconditions),
                                                    (LinearPreconditionData, self.linear_preconditions)]:
                try:
                    parsed_prec = prec_parser(prec)
                    preconditions_list.append(parsed_prec)
                    break
                except:
                    pass
            else:
                raise Exception(f"Couldn't parse precondition: {prec}")

    def parse_effects(self):
        for effect in self._source_action.effects:
            for effect_parser, effects_list in [(SimpleBoolEffectData, self.boolean_effects), (LinearEffectData, self.linear_effects)]:
                try:
                    parsed_effect = effect_parser(effect)
                    effects_list.append(parsed_effect)
                    break
                except:
                    pass
            else:
                raise Exception(f"Couldn't parse effect: {effect}")


class NumericStripsProblem(Problem):
    def __init__(self):
        super().__init__()


def fluent_column_name(fluent, param_idx):
    if fluent == 1:
        return "1"  # constant term, no index needed
    if isinstance(fluent, tuple):
        # fluent with parameters
        return f"{fluent[0]}({','.join('p' + str(i + 1) for i in range(len(fluent[1])))})" + (
            "" if param_idx == 0 else f"#{param_idx + 1}")
    else:
        # fluent with no parameters
        return fluent + ("" if param_idx == 0 else f"#{param_idx + 1}")


class FluentManager:
    def __init__(self, source, parent=None):
        self.source = source
        self.fluent_types = {}
        self.fluent_args_names = {}
        self.fluent_args_types = {}
        self.new_fluents = {}
        self.parent = parent

        for fluent in source.fluents:
            fluent_name = fluent.name
            self.fluent_types[fluent_name] = fluent.type
            args = fluent.signature
            self.fluent_args_names[fluent_name] = [a.name for a in args]
            self.fluent_args_types[fluent_name] = [a.type for a in args]

    def add_fluent(self, fluent_name, fluent_type, fluent_args):
        self.fluent_types[fluent_name] = fluent_type
        self.fluent_args_names[fluent_name] = fluent_args.keys()
        self.fluent_args_types[fluent_name] = fluent_args.values()

    def get_range(self, fluent_name):
        if not self.has(fluent_name) and self.parent is not None:
            return self.parent.get_range(fluent_name)
        fluent_type = self.fluent_types[fluent_name]
        return fluent_type.lower_bound, fluent_type.upper_bound

    def create_fluent(self, fluent_name):
        if fluent_name not in self.fluent_types and self.parent is not None:
            return self.parent.create_fluent(fluent_name)
        if fluent_name in self.new_fluents:
            return self.new_fluents[fluent_name]
        signature = {arg_name: arg_type for (arg_name, arg_type) in zip(self.fluent_args_names[fluent_name],
                                                                        self.fluent_args_types[fluent_name])}
        new_fluent = Fluent(fluent_name, self.fluent_types[fluent_name],
                            **signature)
        self.new_fluents[fluent_name] = new_fluent
        return new_fluent

    def get_possible_parameters(self, fluent_name, parameters=None, objects=None):
        if not fluent_name in self.fluent_types and self.parent is not None:
            return self.parent.get_possible_parameters(fluent_name, parameters, objects)
        possible_parameters = []
        for arg_type in self.fluent_args_types[fluent_name]:
            parameter_options = []
            for p in parameters:
                if p.type == arg_type:
                    parameter_options.append(p)
            for o in objects:
                if o.type == arg_type:
                    parameter_options.append(o)
            possible_parameters.append(parameter_options)
        return possible_parameters

    def create_fluent_expression(self, fluent_name, args_names, parameters=None, objects=None):
        fluent = self.create_fluent(fluent_name)
        if parameters is None:
            parameters = []
        if objects is None:
            objects = []
        parameters = {p.name: p for p in parameters}
        objects = {o.name: o for o in objects}
        args = []
        for a_name in args_names:
            if a_name in parameters:
                args.append(parameters[a_name])
            elif a_name in objects:
                args.append(objects[a_name])
            else:
                raise Exception(f"Couldn't parse argument: {a_name}")
        return fluent(*args)

    def has(self, fluent_name):
        return fluent_name in self.fluent_types

class MultiAgentFluentManager:
    def __init__(self, problem: MultiAgentProblem):
        self.env_fluent_manager = FluentManager(problem.ma_environment)
        self.agents_fluent_managers = {}
        for a in problem.agents:
            self.agents_fluent_managers[a.name] = FluentManager(a, parent=self.env_fluent_manager)

    def is_private(self, fluent_name, agent_name):
        return self.agents_fluent_managers[agent_name].has(fluent_name)


class NumericStripsProblemConverter:
    def __init__(self, problem):

        self.source_problem = problem
        self.numeric_strips_problem = None
        self.source_problem = problem
        self.numeric_variables = []
        self.boolean_variables = []

        self.fluent_manager = None
        self._action_data_getter = {}
        self.fluent_vector_dict = {}

    def _initiate_problem(self):
        self.numeric_strips_problem = Problem()
        self.fluent_manager = FluentManager(self.source_problem)
        self._copy_objects()
        self._copy_actions()

    def owner_prefix(self, action_id):
        return ''

    def generate_linear_expression_fluent_name(self, expression_vector, action_id):
        parts = [f"__{self.owner_prefix(action_id)}__"]
        for col, val in expression_vector.items():
            if val == 0:
                continue
            sign = "P" if val > 0 else "M"
            parts.append(
                f"{sign}{abs(val)}_{col.lower()}__")

        return "".join(parts)

    def get_linear_expression_fluent_args(self, expression_vector, action_id ):
        args = []
        flu_man = self.get_appropriate_fluent_manager(action_id)
        for fluent_name, val in expression_vector.items():
            if val == 0:
                continue
            fluent_name = remove_counter_suffix(fluent_name)
            if fluent_name not in flu_man.fluent_args_types and flu_man.parent is not None:
                arg_types = flu_man.parent.fluent_args_types[fluent_name]
            else:
                arg_types = flu_man.fluent_args_types[fluent_name]
            args += arg_types
        return args

    def get_linear_expression_fluent_range(self, expression_vector, action_id):
        total_lower = 0
        total_upper = 0
        flu_man = self.get_appropriate_fluent_manager(action_id)
        for fluent_name, val in expression_vector.items():
            fluent_name = remove_counter_suffix(fluent_name)
            lower, upper = flu_man.get_range(fluent_name)
            if lower is None:
                total_lower = None
            if total_lower is not None:
                total_lower += min(lower * val, upper * val)
            if upper is None:
                total_upper = None
            if total_upper is not None:
                total_upper += max(lower * val, upper * val)
        return total_lower, total_upper

    def is_private(self, col, agent_name):
        return False

    def environment_fluent_manager(self):
        return self.fluent_manager

    def create_linear_expressions_fluents_and_preconditions(self, preconditions_table):
        for idx, row in preconditions_table.iterrows():
            action_id = row['action_id']
            operator = row['operator']
            value = row['value']
            args = row['args']
            fluent_vector = row.drop(
                ['action_id', 'precondition_index', 'operator', 'value', 'args', 'coeff'])

            new_fluent_name = self.generate_linear_expression_fluent_name(fluent_vector, action_id)
            if not self.environment_fluent_manager().has(new_fluent_name):
                self.fluent_vector_dict[new_fluent_name] = fluent_vector
                lower_bound, higher_bound = self.get_linear_expression_fluent_range(fluent_vector, action_id)
                new_fluent_args = {f'p_{i}': p for i, p in
                                   enumerate(self.get_linear_expression_fluent_args(fluent_vector, action_id))}
                self.environment_fluent_manager().add_fluent(new_fluent_name, RealType(lower_bound, higher_bound), new_fluent_args)
                self.add_fluent_to_ENV(self.environment_fluent_manager().create_fluent(new_fluent_name))
            precon = self.create_precon(action_id, new_fluent_name, args, operator, value)
            if self.action_is_goal(action_id):
                self.add_linear_expression_goal(action_id, precon)

            elif self.action_is_init(action_id):
                continue
            else:
                action = self.get_action(action_id)
                action.add_precondition(precon)

    def get_action(self, action_id):
        if self.action_is_goal(action_id) or self.action_is_init(action_id):
            raise Exception(f"{action_id} is not an action")
        return self.numeric_strips_problem.action(self.action_name(action_id))

    def action_name(self, action_id):
        return action_id

    def create_linear_expressions_effects_and_initial_values(self, effects_table) :
        for idx, row in effects_table.iterrows():
            row = dict(row)
            action_id = row['action_id']
            flu_man = self.get_appropriate_fluent_manager(action_id)
            target_fluent = row['target_fluent']
            target_args = row['target_args']
            change = row['change']
            if self.action_is_goal(action_id):
                continue
            for lin_fluent_name in self.environment_fluent_manager().new_fluents:

                    # For every fluent in the new problem, for every combination of parameters in the new fluent
                    # change the value of this fluent by coeff(new_fluent, old_problem_fluent) * change
                    # If fluent appears numerous times in the new_fluent:
                    # for every appearance, designate it as the target fluent
                    # and treat all other appearances as a regular sub-fluent.

                if not self._linear_fluent_is_affected(lin_fluent_name, target_fluent):
                    continue

                sub_fluent_params_options = self._generate_sub_fluent_parameter_options(action_id, lin_fluent_name)
                vector = self.fluent_vector_dict[lin_fluent_name]

                for i, (sub_fluent, coeff) in enumerate(vector.items()):
                    if remove_counter_suffix(sub_fluent) != target_fluent or coeff == 0:
                        continue
                    params = sub_fluent_params_options.copy()
                    params[i] = [[a] for a in target_args]
                    params = [arg_options for p in params for arg_options in p]
                    for param_combination in itertools.product(*params):
                        changing_fluent = flu_man.create_fluent_expression(lin_fluent_name, param_combination,
                                                               parameters =self.action_parameters(action_id),
                                                               objects = self.numeric_strips_problem.all_objects)
                        new_value = Fraction(str(coeff)) * Fraction(str(change))
                        self.add_linear_effect(action_id, changing_fluent, new_value)

    def _linear_fluent_is_affected(self, lin_fluent_name, fluent_name):
        vector = self.fluent_vector_dict[lin_fluent_name]
        for sub_fluent, _ in vector.items():
            if remove_counter_suffix(sub_fluent) == fluent_name and vector[sub_fluent] != 0:
                return True
        return False

    def get_linear_expression_appropriate_manager(self, lin_fluent_name):
        return self.fluent_manager

    def _generate_sub_fluent_parameter_options(self, action_id, lin_fluent_name):
        vector = self.fluent_vector_dict[lin_fluent_name]
        flu_man = self.get_linear_expression_appropriate_manager(lin_fluent_name)
        sub_fluents_params = []
        for sub_fluent, val in vector.items():
            if val == 0:
                sub_fluents_params.append([])
                continue
            sub_fluent = remove_counter_suffix(sub_fluent)
            possible_parameters = flu_man.get_possible_parameters(sub_fluent,
                                                                  parameters=self.action_parameters(action_id),
                                                                  objects=self.numeric_strips_problem.all_objects)
            possible_parameters = [[p.name for p in p_options] for p_options in possible_parameters]
            sub_fluents_params.append(possible_parameters)
        return sub_fluents_params

    def add_fluent_to_ENV(self, fluent):
        self.numeric_strips_problem.add_fluent(fluent)

    def action_is_goal(self, action_id):
        return action_id == '_GOAL'

    def action_is_init(self, action_id):
        return action_id == '_INIT'

    def action_parameters(self, action_id):
        if self.action_is_goal(action_id) or self.action_is_init(action_id):
            return []
        else:
            return self.get_action(action_id).parameters

    def create_precon(self, action_id, fluent_name, args, operator, value):
        flu_man = self.get_appropriate_fluent_manager(action_id)
        goal = flu_man.create_fluent_expression(fluent_name, args, parameters=self.action_parameters(action_id),
                                                            objects=self.numeric_strips_problem.all_objects)
        precon = get_operator_as_function(operator)(goal, value)
        return precon

    def add_linear_expression_goal(self, action_id, precon):
        self.numeric_strips_problem.add_goal(precon)


    def create_linear_preconditions_table(self):
            columns = [ 'action_id', 'precondition_index', 'operator', 'value', 'args']
            sub_fluent_columns = []
            for action_id, action in self._action_data_getter.items():
                for idx, lin_prec in enumerate(action.linear_preconditions):
                    local_counter = defaultdict(int) # keeps track how many times the fluent_appears in that precondition
                    for sub_fluent in lin_prec.sub_fluents_coeffs:
                        fluent_name, _ = sub_fluent
                        col_name = fluent_column_name(fluent_name, local_counter[fluent_name])
                        local_counter[fluent_name] += 1
                        if col_name not in sub_fluent_columns:
                            sub_fluent_columns.append(col_name)
            columns += sorted(sub_fluent_columns)

            df_precs = pd.DataFrame(columns=columns)
            rows = []
            for action_id, action in self._action_data_getter.items():
                for idx, lin_prec in enumerate(action.linear_preconditions):
                    row = {'action_id': action_id, 'precondition_index': idx,
                               'operator': operator_to_symbol(lin_prec.operator), 'value': lin_prec.value}
                    new_fluent_args = []
                    # Track local occurrences
                    local_counter = defaultdict(int)
                    for fluent, val in lin_prec.sub_fluents_coeffs.items():
                        fluent_name, fluent_args = fluent
                        new_fluent_args += fluent_args
                        count = local_counter[fluent_name]
                        col_name = fluent_column_name(fluent_name, count)
                        row[col_name] = val
                        local_counter[fluent_name] += 1
                        # Fill remaining columns with 0
                    row['args'] = tuple(new_fluent_args)
                    for col in df_precs.columns:
                        if col not in row:
                            row[col] = 0
                    rows.append(row)
            return normalize_dataframe(pd.concat([df_precs, pd.DataFrame(rows)], ignore_index=True),
                                       ignore_as_divisor_columns=['precondition_index', 'value'],
                                       dont_change_columns=['precondition_index'], operator_column='operator')

    def creat_linear_effects_table(self):
        rows = []
        for action_id, action in self._action_data_getter.items():
            for idx, lin_eff in enumerate(action.linear_effects):
                target_fluent, target_args = lin_eff.target_fluent.to_tuple()
                row = {'action_id': action_id, 'effect_index': idx,
                       'target_fluent': target_fluent, 'target_args': target_args, 'change': lin_eff.change}
                rows.append(row)
        return pd.DataFrame(rows)

    def parse_goal_conditions(self):
        raise NotImplementedError

    def parse_initial_values(self):
        raise NotImplementedError

    def compile(self):
        self._initiate_problem()

        self.parse_goal_conditions()
        self.parse_initial_values()

        for action_id, action in self._action_data_getter.items():
            if not self._is_action(action_id):
                continue
            action.parse_preconditions()
            action.parse_effects()

        linear_preconditions_table = self.create_linear_preconditions_table()
        linear_effects_table =  self.creat_linear_effects_table()

        for action_id, action_data in self._action_data_getter.items():
            for effect in action_data.boolean_effects:
                fluent, val = effect.target_fluent, effect.value
                self.add_boolean_effect(action_id, fluent, val)
            for prec in action_data.boolean_preconditions:
                fluent = prec.target_fluent
                self.add_boolean_precondition(action_id, fluent)

        self.create_linear_expressions_fluents_and_preconditions(linear_preconditions_table)
        self.create_linear_expressions_effects_and_initial_values(linear_effects_table)
        return self.numeric_strips_problem

    def _is_action(self, action_id):
        return not (self.action_is_init(action_id) or self.action_is_goal(action_id))

    def _copy_objects(self):
        for obj in self.source_problem.all_objects:
            self.numeric_strips_problem.add_object(Object(obj.name, obj.type))

    def _copy_actions(self):
        for action in self.source_problem.actions:
            new_action_data = NumericStripsActionData(action)
            new_action = InstantaneousAction(new_action_data.name, *new_action_data.name.parameters)
            self.numeric_strips_problem.add_action(new_action)
            self._action_data_getter[action.name] = new_action_data

    def get_appropriate_fluent_manager(self, action_id):
        return self.fluent_manager

    def add_boolean_effect(self, action_id, fluent: AtomicFluent, val):
        action_name = action_id
        fluent_name, fluent_args = fluent.to_tuple()
        flu_man = self.fluent_manager
        if action_name != '_INIT':
            action = self.numeric_strips_problem.actions(action_name)
            fluent_expr = flu_man.create_fluent_expression(fluent_name, fluent_args, action.parameters,
                                                               self.numeric_strips_problem.all_objects)
            action.add_effect(fluent_expr, val)
        else:
            flu_man = self.fluent_manager
            fluent_expr = flu_man.create_fluent_expression(fluent_name, fluent_args, [],
                                                           self.numeric_strips_problem.all_objects)
            self.numeric_strips_problem.set_initial_value(fluent_expr, val)

    def add_boolean_precondition(self, action_id, fluent: AtomicFluent):
        action_name = action_id
        fluent_name, fluent_args = fluent.to_tuple()
        flu_man = self.fluent_manager
        if action_name != '_GOAL':
            action = self.numeric_strips_problem.action(action_name)
            fluent_expr = flu_man.create_fluent_expression(fluent_name, fluent_args, action.parameters,
                                                               self.numeric_strips_problem.all_objects)
            action.add_precondition(fluent_expr)
        else:
            fluent_expr = flu_man.create_fluent_expression(fluent_name, fluent_args, [],
                                                               self.numeric_strips_problem.all_objects)
            self.numeric_strips_problem.add_public_goal(fluent_expr)

    def add_linear_effect(self, action_id, fluent_expr, new_value):
        if self.action_is_init(action_id):
            try:
                curr_v = Fraction(str(self.numeric_strips_problem.initial_value(fluent_expr)))
            except:
                curr_v = 0
            adjusted_v = curr_v + new_value
            self.numeric_strips_problem.set_initial_value(fluent_expr, adjusted_v)
        else:
            self.get_action(action_id).add_effect(fluent_expr, Plus(fluent_expr, new_value))


class MultiAgentNumericStripsProblemConverter(NumericStripsProblemConverter):

    def __init__(self, problem):
        NumericStripsProblemConverter.__init__(self, problem)

    def get_linear_expression_appropriate_manager(self, lin_fluent_name):
        agent_name = lin_fluent_name.split('__')[1]
        if agent_name in self.fluent_manager.agents_fluent_managers:
            return self.fluent_manager.agents_fluent_managers[agent_name]
        return self.fluent_manager.env_fluent_manager

    def add_linear_expression_goal(self, action_id, precon):
        agent_name, _ = action_id
        self.numeric_strips_problem.agent(agent_name).add_public_goal(precon)

    def action_is_goal(self, action_id):
        _, action_name = action_id
        return action_name == '_GOAL'

    def action_is_init(self, action_id):
        _, action_name = action_id
        return action_name == '_INIT'

    def add_fluent_to_ENV(self, fluent):
        self.numeric_strips_problem.ma_environment.add_fluent(fluent)

    def environment_fluent_manager(self):
        return self.fluent_manager.env_fluent_manager

    def owner_prefix(self, action_id):
        return action_id[0]

    def action_name(self, action_id):
        return action_id[1]

    def get_action(self, action_id):
        if self.action_is_goal(action_id) or self.action_is_init(action_id):
            raise Exception(f"{action_id} is not an action")
        return self.numeric_strips_problem.agent(action_id[0]).action(self.action_name(action_id))

    def get_appropriate_fluent_manager(self, action_id):
        agent_name, _ = action_id
        if agent_name != '_ENV':
            return self.fluent_manager.agents_fluent_managers[agent_name]
        else:
            return self.fluent_manager.env_fluent_manager

    def add_boolean_effect(self, action_id, fluent:AtomicFluent, val):
        agent_name, action_name = action_id
        fluent_name, fluent_args = fluent.to_tuple()
        if agent_name != '_ENV':
            agent = self.numeric_strips_problem.agent(agent_name)
            flu_man = self.fluent_manager.agents_fluent_managers[agent_name]
            if action_name != '_INIT':
                action = agent.action(action_name)
                fluent_expr = flu_man.create_fluent_expression(fluent_name, fluent_args, action.parameters,
                                                 self.numeric_strips_problem.all_objects)
                action.add_effect(fluent_expr, val)
            else:
                fluent_expr = flu_man.create_fluent_expression(fluent_name, fluent_args, [],
                                                               self.numeric_strips_problem.all_objects)
                self.numeric_strips_problem.set_initial_value(Dot(agent, fluent_expr), val)
        else:
            flu_man = self.fluent_manager.env_fluent_manager
            fluent_expr = flu_man.create_fluent_expression(fluent_name, fluent_args, [],
                                                           self.numeric_strips_problem.all_objects)
            self.numeric_strips_problem.set_initial_value(fluent_expr, val)

    def add_boolean_precondition(self, action_id, fluent:AtomicFluent):
        agent_name, action_name = action_id

        fluent_name, fluent_args = fluent.to_tuple()
        if agent_name != '_ENV':
            agent = self.numeric_strips_problem.agent(agent_name)
            flu_man = self.fluent_manager.agents_fluent_managers[agent_name]
            if action_name != '_GOAL':
                action = agent.action(action_name)
                fluent_expr = flu_man.create_fluent_expression(fluent_name, fluent_args, action.parameters,
                                                               self.numeric_strips_problem.all_objects)
                action.add_precondition(fluent_expr)
            else:
                fluent_expr = flu_man.create_fluent_expression(fluent_name, fluent_args, [],
                                                               self.numeric_strips_problem.all_objects)
                agent.add_public_goal(fluent_expr)



    def parse_goal_conditions(self):
        for agent in self.source_problem.agents:
            goal_data = NumericStripsActionData(None)
            goal_data.name = '_GOAL'
            for goal in agent.public_goals:
                for prec_parser, preconditions_list in [(SimpleBoolPreconditionData, goal_data.boolean_preconditions),
                                                            (LinearPreconditionData, goal_data.linear_preconditions),]:
                    try:
                        parsed_prec = prec_parser(goal)
                        preconditions_list.append(parsed_prec)
                        break
                    except:
                        pass
                else:
                    raise Exception(f"Couldn't parse goal condition: {goal}")
            self._action_data_getter[(agent.name, '_GOAL')] = goal_data

    def parse_initial_values(self):
        init_values_data = NumericStripsActionData(None)
        init_values_data.name = '_INIT'

        init_vals = self.source_problem.initial_values
        for fnode, val in init_vals.items():
            str_fnode = str(fnode)
            if '.' in str_fnode:
                agent_name = str_fnode.split('.')[0]
                fnode = fnode.args[0]
            else:
                agent_name = '_ENV'

            if not (agent_name, '_INIT') in self._action_data_getter:
                self._action_data_getter[(agent_name, '_INIT')] = NumericStripsActionData(None)
            init_values_data = self._action_data_getter[(agent_name, '_INIT')]
            for effect_parser, effects_list in [(SimpleBoolEffectData.from_init_val, init_values_data.boolean_effects),
                                                        (LinearEffectData.from_init_val, init_values_data.linear_effects), ]:
                try:
                    parsed_value = effect_parser(fnode, val)
                    effects_list.append(parsed_value)
                    break
                except:
                    pass
            else:
                raise Exception(f"Couldn't parse init value: {fnode, val}")

    def _initiate_problem(self):
        self.numeric_strips_problem = MultiAgentProblem()
        self.fluent_manager = MultiAgentFluentManager(self.source_problem)
        self._copy_agents()
        for agent, flu_man in zip(self.numeric_strips_problem.agents, self.fluent_manager.agents_fluent_managers.values()):
            for fluent_name in flu_man.fluent_types:
                agent.add_fluent(flu_man.create_fluent(fluent_name))

        self._copy_objects()
        self._copy_actions()


    def _copy_actions(self):
        for source_agent in self.source_problem.agents:
            for source_action in source_agent.actions:
                agent = self.numeric_strips_problem.agent(source_agent.name)
                new_action_data = NumericStripsActionData(source_action)
                new_action = InstantaneousAction(new_action_data.name, **new_action_data.parameters)
                self.numeric_strips_problem.agent(source_agent.name).add_action(new_action)
                self._action_data_getter[(agent.name, source_action.name)] = new_action_data

    def _copy_agents(self):
        for agent in self.source_problem.agents:
            self.numeric_strips_problem.add_agent(Agent(agent.name, self.numeric_strips_problem))
