from itertools import chain

from unified_planning.engines.results import CompilerResult
from unified_planning.engines.compilers.utils import replace_action
from functools import partial

from unified_planning.model.multi_agent import Agent
from unified_planning.shortcuts import *
from unified_planning.model import InstantaneousAction, Action
from typing import Dict

import up_social_laws
from up_social_laws.ma_problem_waitfor import MultiAgentProblemWithWaitfor
from up_social_laws.robustness_verification import InstantaneousActionRobustnessVerifier, FluentMap, RobustnessVerifier
from up_social_laws.snp_to_num_strips import MultiAgentNumericStripsProblemConverter


class SimpleNumericRobustnessVerifier(InstantaneousActionRobustnessVerifier):

    def __init__(self):
        RobustnessVerifier.__init__(self)
        self.og_problem = None

    @property
    def name(self):
        return "snrbv"

    @staticmethod
    def supported_kind() -> ProblemKind:
        supported_kind = unified_planning.model.problem_kind.multi_agent_kind.union(
            unified_planning.model.problem_kind.actions_cost_kind)
        return supported_kind


    def create_action_copy(self, problem: MultiAgentProblemWithWaitfor, agent: Agent, action: InstantaneousAction,
                           prefix: str):
        """
        Create a new copy of an action, with name prefix_action_name, and duplicates the local preconditions/effects
        """
        d = {}
        for p in action.parameters:
            d[p.name] = p.type

        new_action = InstantaneousAction(
            up_social_laws.name_separator.join([prefix, agent.name, action.name]), _parameters=d)
        for fact in self.get_action_preconditions(problem, agent, action, True, True):
            new_action.add_precondition(self.fsub.substitute(fact, self.local_fluent_map[agent], agent))
        for effect in action.effects:
            new_fluent, new_value = self.substitute_effect(effect, self.local_fluent_map[agent], agent)
            new_action.add_effect(new_fluent, new_value)

        return new_action

    def substitute_effect(self, effect: Effect, fmap: FluentMap, local_agent: Agent):
        fluent = effect.fluent
        new_fluent = self.fsub.substitute(fluent, fmap, local_agent)
        args = effect.value.args
        new_args = []
        for arg in args:
            if arg.node_type == OperatorKind.FLUENT_EXP:
                new_args.append(self.fsub.substitute(arg, fmap, local_agent))
            else:
                new_args.append(arg)

        args_sub = {args[i]: new_args[i] for i, _ in enumerate(new_args)}
        new_value = effect.value.substitute(args_sub)
        return new_fluent, new_value

    def get_success_action(self, agent, action):
        a_s = self.create_action_copy(self.og_problem, agent, action, f"s")
        a_s.add_precondition(self.act)
        a_s.add_precondition(Not(self.agent_wt[agent.name]))
        a_s.add_precondition(Not(self.cf))
        for pre in self.get_action_preconditions(self.og_problem, agent, action, True, True):
            a_s.add_precondition(self.fsub.substitute(pre, self.global_fluent_map, agent))

        for eff in action.effects:
            a_s.add_effect(*self.substitute_effect(eff, self.global_fluent_map, agent))
            if str(eff.value) != 'true' or eff.fluent not in self.prec_wt:
                continue
            a_s.add_precondition(Not(self.prec_wt[eff.fluent]))

        for _, eff in self.linear_effects_table.iterrows():
            eff = eff.to_dict()
            v = eff['target_fluent'], eff['target_args']
            if v[0] in agent.fluents:
                fluent = agent.fluent(v[0])
            else:
                fluent = self.og_problem.ma_environment.fluent(v[0])
            args = [self.og_problem.object(o) for o in v[1]]
            v_fluent = fluent(*args)
            action_id = (agent.name, action.name)
            if action_id != eff['action_id']:
                continue
            k = eff['change']
            for w0 in self.W_v[v]:
                if not GE(v_fluent, w0) in self.prec_wt:
                    print('Warning, weird things with wt^phi happen')
                    continue
                v_lt_w0_minus_k = self.fsub.substitute(LT(v_fluent, Minus(w0, k)), self.global_fluent_map, agent)
                a_s.add_precondition(Or(Not(self.prec_wt[GE(v_fluent, w0)]), v_lt_w0_minus_k))
        return a_s

    def get_fail_action(self, agent, action, failed_prec):
        a_f = self.create_action_copy(self.og_problem, agent, action, f"f_NOT_{failed_prec}")
        a_f.add_precondition(self.act)
        a_f.add_precondition(Not(self.agent_wt[agent.name]))
        a_f.add_precondition(Not(self.cf))
        for pre in self.get_action_preconditions(self.og_problem, agent, action, False, True):
            a_f.add_precondition(self.fsub.substitute(pre, self.global_fluent_map, agent))
        a_f.add_precondition(Not(self.fsub.substitute(failed_prec, self.global_fluent_map, agent)))
        a_f.add_effect(self.fail, True)
        a_f.add_effect(self.cf, True)
        return a_f

    def get_wait_action(self, agent, action, wt_prec):
        a_wt = self.create_action_copy(self.og_problem, agent, action, f"wt_{wt_prec}")
        a_wt.add_precondition(self.act)
        a_wt.add_precondition(Not(self.agent_wt[agent.name]))
        a_wt.add_precondition(Not(self.cf))
        for pre in self.get_action_preconditions(self.og_problem, agent, action, True, True):
            if pre == wt_prec:
                continue
            a_wt.add_precondition(self.fsub.substitute(pre, self.global_fluent_map, agent))
        a_wt.add_precondition(Not(self.fsub.substitute(wt_prec, self.global_fluent_map, agent)))

        a_wt.add_effect(self.prec_wt[wt_prec], True)
        a_wt.add_effect(self.fail, True)
        a_wt.add_effect(self.agent_wt[agent.name], True)

        a_wt.add_effect(self.cf, False)
        return a_wt

    def get_local_action_cf(self, agent, action):
        a_l_cf = self.create_action_copy(self.og_problem, agent, action, f"l_cf")
        a_l_cf.add_precondition(self.act)
        a_l_cf.add_precondition(self.cf)
        return a_l_cf

    def get_local_action_wt(self, agent, action):
        a_l_wt = self.create_action_copy(self.og_problem, agent, action, f"l_wt")
        a_l_wt.add_precondition(self.act)
        a_l_wt.add_precondition(self.agent_wt[agent.name])
        return a_l_wt

    def _compile(self, problem: "up.model.AbstractProblem",
                 compilation_kind: "up.engines.CompilationKind") -> CompilerResult:
        """
        Creates a robustness verification problem.
        """

        # Represents the map from the new action to the old action
        new_to_old: Dict[Action, Action] = {}

        self.og_problem = problem

        new_problem = self.initialize_problem(self.og_problem)

        # Setting up auxiliary atoms
        self.act = Fluent('act', BoolType())
        self.fail = Fluent('fail', BoolType())
        self.cf = Fluent('cf', BoolType())

        self.fin = {}
        self.agent_wt = {}

        self.W_v = {}
        num_con = MultiAgentNumericStripsProblemConverter(problem)
        self.linear_preconditions_table = num_con.create_linear_preconditions_table()
        self.linear_effects_table = num_con.create_linear_effects_table()

        for v in self.linear_preconditions_table.columns:
            if v in ['action_id', 'precondition_index', 'operator', 'value', 'args', 'coeff']:
                continue
            for _, row in self.linear_preconditions_table.iterrows():
                row = row.to_dict()
                args = row['args']
                assert row[v] == 1 and row['operator'] == '>='
                if (v, args) not in self.W_v:
                    self.W_v[(v, args)] = []
                self.W_v[v, args].append(row['value'])

        for a in self.og_problem.agents:
            self.fin[a.name] = Fluent(f"fin_{a.name}")
            self.agent_wt[a.name] = Fluent(f"wt_{a.name}")

            new_problem.add_fluent(self.fin[a.name], default_initial_value=False)
            new_problem.add_fluent(self.agent_wt[a.name], default_initial_value=False)

        self.waitfor_precs = {}
        for agent, action in chain(*[[(agent, action) for action in agent.actions] for agent in problem.agents]):
            self.waitfor_precs[(agent.name, action.name)] = self.get_action_preconditions(problem, agent, action, False, True)
        self.all_waitfor_precs = chain(*list(self.waitfor_precs.values()))

        self.prec_wt = {}
        for prec in self.all_waitfor_precs:
            if prec in self.prec_wt:
                continue
            self.prec_wt[prec] = Fluent("wt_" + str(prec))
            new_problem.add_fluent(self.prec_wt[prec], default_initial_value=False)

        new_problem.add_fluent(self.act, default_initial_value=True)
        new_problem.add_fluent(self.fail, default_initial_value=False)
        new_problem.add_fluent(self.cf, default_initial_value=False)

        for agent in self.og_problem.agents:
            for action in agent.actions:

                # Fail version
                for prec in self.get_action_preconditions(problem, agent, action, True, False):
                    a_f = self.get_fail_action(agent, action, prec)
                    new_problem.add_action(a_f)
                    new_to_old[a_f] = action

                # local version
                a_local_cf = self.get_local_action_cf(agent, action)
                new_problem.add_action(a_local_cf)
                new_to_old[a_local_cf] = action

                # local version
                a_local_wt = self.get_local_action_wt(agent, action)
                new_problem.add_action(a_local_wt)
                new_to_old[a_local_wt] = action

                # success version
                a_succ = self.get_success_action(agent, action)
                new_problem.add_action(a_succ)
                new_to_old[a_succ] = action

                for prec in problem.waitfor.waitfor_map.get((agent.name, action.name), []):
                    a_w = self.get_wait_action(agent, action, prec)
                    new_problem.add_action(a_w)
                    new_to_old[a_w] = action

            # end-success
            end_s = InstantaneousAction(f"end_s_{agent.name}")
            end_s.add_precondition(Not(self.fin[agent.name]))
            for goal in self.get_agent_goal(problem, agent):
                end_s.add_precondition(self.fsub.substitute(goal, self.global_fluent_map, agent))
                end_s.add_precondition(self.fsub.substitute(goal, self.local_fluent_map[agent], agent))
            end_s.add_effect(self.fin[agent.name], True)
            end_s.add_effect(self.act, False)

            new_problem.add_action(end_s)

            end_f = InstantaneousAction(f"end_f_{agent.name}")
            end_f.add_precondition(Not(self.fin[agent.name]))
            end_f.add_effect(self.act, False)
            global_goal_prec = None
            for goal in self.get_agent_goal(problem, agent):
                not_g_g = Not(self.fsub.substitute(goal, self.global_fluent_map, agent))
                if global_goal_prec is None:
                    global_goal_prec = not_g_g
                else:
                    global_goal_prec = Or(not_g_g, global_goal_prec)
                end_f.add_precondition(self.fsub.substitute(goal, self.local_fluent_map[agent], agent))
            end_f.add_precondition(global_goal_prec)
            end_f.add_effect(self.fin[agent.name], True)
            end_f.add_effect(self.fail, True)
            new_problem.add_action(end_f)

        # Goal
        new_problem.add_goal(self.fail)
        for fin in self.fin.values():
            new_problem.add_goal(fin)

        return CompilerResult(
            new_problem, partial(replace_action, map=new_to_old), self.name
        )
