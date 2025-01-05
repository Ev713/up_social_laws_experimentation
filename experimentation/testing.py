import json
import random

import unified_planning
from unified_planning.model import InstantaneousAction, Fluent
from unified_planning.model.multi_agent import Agent
from unified_planning.shortcuts import *

from experimentation import simulate
from up_social_laws.ma_problem_waitfor import MultiAgentProblemWithWaitfor
from up_social_laws.social_law import SocialLaw


problem = Problem()
fluent = Fluent('fluent', BoolType(), p=IntType(0, 100))
a = InstantaneousAction('action', )
a.add_effect(fluent(1), True)
problem.add_action(a)
problem.add_fluent(fluent, default_initial_value=False)

simulate(problem)

