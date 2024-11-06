from unified_planning.engines import CompilerResult

import unified_planning as up
import unified_planning.engines as engines
from unified_planning.engines.mixins.compiler import CompilationKind, CompilerMixin
from unified_planning.model.multi_agent import *
from unified_planning.model import *
from unified_planning.engines.results import CompilerResult
from unified_planning.exceptions import UPExpressionDefinitionError, UPProblemDefinitionError
from typing import List, Dict, Union, Optional
from unified_planning.engines.compilers.utils import replace_action, get_fresh_name
from functools import partial
from operator import neg
from unified_planning.model import Parameter, Fluent, InstantaneousAction
from unified_planning.shortcuts import *
from unified_planning.exceptions import UPProblemDefinitionError
from unified_planning.model import Problem, InstantaneousAction, DurativeAction, Action
from typing import List, Dict
from itertools import product
from up_social_laws.waitfor_specification import WaitforSpecification
from up_social_laws.ma_problem_waitfor import MultiAgentProblemWithWaitfor
import unified_planning as up
from unified_planning.engines import Credits
from unified_planning.io.pddl_writer import PDDLWriter
import unified_planning.model.walkers as walkers
from unified_planning.model.walkers.identitydag import IdentityDagWalker
from unified_planning.environment import get_environment
import unified_planning.model.problem_kind
import up_social_laws
from up_social_laws.robustness_verification import InstantaneousActionRobustnessVerifier, FluentMap, RobustnessVerifier, \
    FluentMapSubstituter


class RegularWaitingActionRobustnessVerifier(RobustnessVerifier):
    '''Robustness verifier class for instantaneous actions using alternative formulation:
    this class requires a (multi agent) problem, and creates a classical planning problem which is unsolvable iff the multi agent problem is not robust.
    Implements the robustness verification compilation from Tuisov, Shleyfman, Karpas with the bugs fixed
    '''

    def __init__(self):
        RobustnessVerifier.__init__(self)

    @property
    def name(self):
        return "wrbv"

    @staticmethod
    def supported_kind() -> ProblemKind:
        supported_kind = ProblemKind()
        supported_kind = unified_planning.model.problem_kind.multi_agent_kind.union(
            unified_planning.model.problem_kind.actions_cost_kind)
        return supported_kind

    @staticmethod
    def supports(problem_kind):
        return problem_kind <= InstantaneousActionRobustnessVerifier.supported_kind()

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

    def create_action_copy(self, problem: MultiAgentProblemWithWaitfor, agent: Agent, action: InstantaneousAction,
                           prefix: str):
        """Create a new copy of an action, with name prefix_action_name, and duplicates the local preconditions/effects
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

    def get_succes_action(self, agent, action, all_actions_allowed):
        prefix = 'ra' if all_actions_allowed else 'a'
        a_s = self.create_action_copy(self.og_problem, agent, action, f"s_{prefix}_")
        a_s.add_precondition(self.stage_1)
        allow_action = self.allow_action_map[agent.name][action.name](*action.parameters)
        restrict_actions = self.restrict_actions_map[agent.name]
        if all_actions_allowed:
            a_s.add_precondition(Not(restrict_actions))
        else:
            a_s.add_precondition(allow_action)
        for fact in self.get_action_preconditions(self.og_problem, agent, action, True, True):
            a_s.add_precondition(self.fsub.substitute(fact, self.global_fluent_map, agent))
        for effect in action.effects:
            a_s.add_effect(*self.substitute_effect(effect, self.global_fluent_map, agent))
        a_s.add_effect(allow_action, False)
        a_s.add_effect(restrict_actions, False)
        return a_s

    def get_fail_action(self, agent, action, fact, i, all_actions_allowed):
        prefix = 'ra' if all_actions_allowed else 'a'
        a_f = self.create_action_copy(self.og_problem, agent, action, f"f_{prefix}_{i}")
        if all_actions_allowed:
            restrict_actions = self.restrict_actions_map[agent.name]
            a_f.add_precondition(Not(restrict_actions))
        else:
            allow_action = self.allow_action_map[agent.name][action.name](*action.parameters)
            a_f.add_precondition(allow_action)
        a_f.add_precondition(self.stage_1)
        for pre in self.get_action_preconditions(self.og_problem, agent, action, False, True):
            a_f.add_precondition(self.fsub.substitute(pre, self.global_fluent_map, agent))
        a_f.add_precondition(Not(self.fsub.substitute(fact, self.global_fluent_map, agent)))
        a_f.add_effect(self.precondition_violation, True)
        a_f.add_effect(self.stage_2, True)
        a_f.add_effect(self.stage_1, False)
        return a_f

    def get_wait_action(self, agent, action, fact, i, all_actions_allowed):
        prefix = 'ra' if all_actions_allowed else 'a'
        a_w = self.create_action_copy(self.og_problem, agent, action, f"w_{prefix}_{i}")
        allow_action = self.allow_action_map[agent.name][action.name](*action.parameters)
        restrict_actions = self.restrict_actions_map[agent.name]
        if all_actions_allowed:
            a_w.add_precondition(Not(restrict_actions))
        else:
            a_w.add_precondition(allow_action)
        a_w.add_precondition(self.stage_1)
        a_w.add_precondition(Not(self.fsub.substitute(fact, self.global_fluent_map, agent)))
        assert not fact.is_not()
        a_w.clear_effects()
        a_w.add_effect(restrict_actions, True)
        a_w.add_effect(allow_action, True)
        return a_w

    def get_deadlock_action(self, agent, action, fact, i):
        a_deadlock = self.create_action_copy(self.og_problem, agent, action, f"d{i}")
        a_deadlock.clear_preconditions()
        a_deadlock.add_precondition(Not(self.fsub.substitute(fact, self.global_fluent_map, agent)))
        allow_action = self.allow_action_map[agent.name][action.name](*action.parameters)
        restrict_actions = self.restrict_actions_map[agent.name]
        a_deadlock.add_precondition(restrict_actions)
        a_deadlock.add_precondition(allow_action)

        a_deadlock.clear_effects()
        a_deadlock.add_effect(self.fin(self.get_agent_obj(agent)), True)
        a_deadlock.add_effect(self.possible_deadlock, True)
        a_deadlock.add_effect(self.stage_1, False)
        return a_deadlock

    def get_local_action(self, agent, action, all_actions_allowed):
        prefix = 'ra' if all_actions_allowed else 'a'
        a_local = self.create_action_copy(self.og_problem, agent, action, f"l_{prefix}")
        a_local.add_precondition(self.stage_2)
        a_local.add_precondition(self.agent_turn_map[agent.name])
        allow_action = self.allow_action_map[agent.name][action.name](*action.parameters)
        restrict_actions = self.restrict_actions_map[agent.name]
        if all_actions_allowed:
            a_local.add_precondition(Not(restrict_actions))
        else:
            a_local.add_precondition(allow_action)
        a_local.add_effect(allow_action, False)
        a_local.add_effect(restrict_actions, False)
        return a_local

    def _compile(self, problem: "up.model.AbstractProblem",
                 compilation_kind: "up.engines.CompilationKind") -> CompilerResult:
        """
        Creates a robustness verification problem.
        """

        # Represents the map from the new action to the old action
        new_to_old: Dict[Action, Action] = {}

        self.og_problem = problem

        new_problem = self.initialize_problem(self.og_problem)

        self.waiting_fluent_map = FluentMap("w", default_value=False)
        self.waiting_fluent_map.add_facts(self.og_problem, new_problem)
        self.allow_action_map = {}
        self.restrict_actions_map = {}
        self.agent_turn_map = {}

        # Add fluents
        self.stage_1 = Fluent("stage-1")
        self.stage_2 = Fluent("stage-2")
        self.precondition_violation = Fluent("precondition-violation")
        self.possible_deadlock = Fluent("possible-deadlock")
        self.conflict = Fluent("conflict")
        self.fin = Fluent("fin", _signature=[Parameter("a", self.agent_type)])

        new_problem.add_fluent(self.stage_1, default_initial_value=False)
        new_problem.add_fluent(self.stage_2, default_initial_value=False)
        new_problem.add_fluent(self.precondition_violation, default_initial_value=False)
        new_problem.add_fluent(self.possible_deadlock, default_initial_value=False)
        new_problem.add_fluent(self.conflict, default_initial_value=False)
        new_problem.add_fluent(self.fin, default_initial_value=False)

        for agent_id, agent in enumerate(self.og_problem.agents):
            agent_turn = Fluent(f'{agent.name}_turn', BoolType())
            self.agent_turn_map[agent.name] = agent_turn
            if agent_id == 0:
                new_problem.add_fluent(agent_turn, default_initial_value=True)
            else:
                new_problem.add_fluent(agent_turn, default_initial_value=False)
                take_turn = InstantaneousAction(f'{agent.name}_take_turn')
                take_turn.add_precondition(self.stage_2)
                prev_turn = self.agent_turn_map[self.og_problem.agents[agent_id - 1].name]
                take_turn.add_precondition(prev_turn)
                take_turn.add_effect(agent_turn, True)
                take_turn.add_effect(prev_turn, False)
                new_problem.add_action(take_turn)

            restrict_actions = Fluent(f'restrict_actions_{agent.name}', BoolType())
            self.restrict_actions_map[agent.name] = restrict_actions
            new_problem.add_fluent(restrict_actions, default_initial_value=False)
            for action in agent.actions:
                signature = {f'p{i}': p.type for i, p in enumerate(action.parameters)}
                action_fluent = Fluent("allow-" + agent.name + "-" + action.name, BoolType(), **signature)
                # allow_action_map.setdefault(action.agent, {}).update(action=action_fluent)
                if agent.name not in self.allow_action_map.keys():
                    self.allow_action_map[agent.name] = {action.name: action_fluent}
                else:
                    self.allow_action_map[agent.name][action.name] = action_fluent
                new_problem.add_fluent(action_fluent, default_initial_value=False)

        # Add actions
        for agent in self.og_problem.agents:
            for action in agent.actions:
                # Success version - affects globals same way as original
                for all_actions_allowed in [True, False]:
                    a_s = self.get_succes_action(agent, action, all_actions_allowed)
                    new_problem.add_action(a_s)
                    new_to_old[a_s] = action

                # Fail version
                for i, fact in enumerate(self.get_action_preconditions(self.og_problem, agent, action, True, False)):
                    for all_actions_allowed in [True, False]:
                        a_f = self.get_fail_action(agent, action, fact, i, all_actions_allowed)
                        new_problem.add_action(a_f)
                        new_to_old[a_f] = action

                for i, fact in enumerate(self.get_action_preconditions(self.og_problem, agent, action, False, True)):
                    # Wait version
                    for all_actions_allowed in [True, False]:
                        a_w = self.get_wait_action(agent, action, fact, i, all_actions_allowed)
                        new_problem.add_action(a_w)
                        new_to_old[a_w] = action

                    # deadlock version
                    a_deadlock = self.get_deadlock_action(agent, action, fact, i)
                    new_problem.add_action(a_deadlock)
                    new_to_old[a_deadlock] = action

                # local version
                for all_actions_allowed in [True, False]:
                    a_local = self.get_local_action(agent, action, all_actions_allowed)
                    new_problem.add_action(a_local)
                    new_to_old[a_local] = action

            # end-success
            end_s = InstantaneousAction(f"end_s_{agent.name}")
            for goal in self.get_agent_goal(self.og_problem, agent):
                end_s.add_precondition(self.fsub.substitute(goal, self.global_fluent_map, agent))
            end_s.add_effect(self.fin(self.get_agent_obj(agent)), True)
            end_s.add_effect(self.stage_1, False)
            new_problem.add_action(end_s)
            new_to_old[end_s] = None

        # start-stage-2
        start_stage_2 = InstantaneousAction("start_stage_2")
        for agent in self.og_problem.agents:
            start_stage_2.add_precondition(self.fin(self.get_agent_obj(agent)))
        start_stage_2.add_effect(self.stage_2, True)
        start_stage_2.add_effect(self.stage_1, False)
        new_problem.add_action(start_stage_2)
        new_to_old[start_stage_2] = None

        for agent in self.og_problem.agents:
            for i, goal in enumerate(self.get_agent_goal(self.og_problem, agent)):
                goals_not_achieved = InstantaneousAction(f"goals_not_achieved_{agent.name}_{i}")
                goals_not_achieved.add_precondition(self.stage_2)
                goals_not_achieved.add_precondition(Not(self.fsub.substitute(goal, self.global_fluent_map, agent)))
                for a in self.og_problem.agents:
                    for g in self.get_agent_goal(self.og_problem, a):
                        goals_not_achieved.add_precondition(self.fsub.substitute(g, self.local_fluent_map[agent], agent))
                goals_not_achieved.add_effect(self.conflict, True)
                new_problem.add_action(goals_not_achieved)
                new_to_old[goals_not_achieved] = None

        # declare_deadlock
        declare_deadlock = InstantaneousAction("declare_deadlock")
        declare_deadlock.add_precondition(self.stage_2)
        declare_deadlock.add_precondition(self.possible_deadlock)
        for agent in self.og_problem.agents:
            for goal in self.get_agent_goal(self.og_problem, agent):
                declare_deadlock.add_precondition(self.fsub.substitute(goal, self.local_fluent_map[agent], agent))
        declare_deadlock.add_effect(self.conflict, True)
        new_problem.add_action(declare_deadlock)
        new_to_old[declare_deadlock] = None

        # declare_fail
        declare_fail = InstantaneousAction("declare_fail")
        declare_fail.add_precondition(self.stage_2)
        declare_fail.add_precondition(self.precondition_violation)
        for agent in self.og_problem.agents:
            for goal in self.get_agent_goal(self.og_problem, agent):
                declare_fail.add_precondition(self.fsub.substitute(goal, self.local_fluent_map[agent], agent))
        declare_fail.add_effect(self.conflict, True)
        new_problem.add_action(declare_fail)
        new_problem.set_initial_value(self.stage_1, True)
        new_to_old[declare_fail] = None

        # Goal
        new_problem.add_goal(self.conflict)

        return CompilerResult(
            new_problem, partial(replace_action, map=new_to_old), self.name
        )


class NumericWaitingActionRobustnessVerifier(RegularWaitingActionRobustnessVerifier):
    pass
