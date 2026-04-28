import unified_planning
from unified_planning.model import Fluent, InstantaneousAction
from unified_planning.model.multi_agent import Agent
from unified_planning.shortcuts import (UserType, BoolType, Equals, And, Not, IntType, GE, LE, Minus, Plus, GT, 
                                        RealType, LT, Times, Dot)

from experimentation.problem_generators.problem_generator import ProblemGenerator, NumericProblemGenerator
from up_social_laws.ma_problem_waitfor import MultiAgentProblemWithWaitfor
from up_social_laws.social_law import SocialLaw


class NumericCivGenerator(NumericProblemGenerator):
    """
    Multi-agent numeric problem generator for the Civilization domain.
    
    Multi-agent decomposition:
    - Agent-specific fluents: train/boat cargo state, ownership, carts_at (per agent)
    - Global/Environmental fluents: available(r, p) - SHARED resources at each location,
      building properties, location connections, pollution, labour, resource-use, housing

    Key design: resources are shared at locations, creating natural agent interference.
    The social law for this domain only adds per-agent resource accounting and waitfor
    annotations for train/boat movement capacity preconditions, plus boat end-state rules.
    """
    
    def __init__(self):
        super().__init__()
        self.agent_type_name = 'agent'
        self.has_boats = False
        self.has_trains = False
        self.bound_aliases = {
            'resources-in-boat': 'boat-load',
            'boat-space-in': 'boat-load',
            'resources-in-train': 'train-load',
            'train-space-in': 'train-load',
        }
        self.default_bounds = {
            'boat-capacity': 2,
            'train-capacity': 2,
            'boat-load': 3,
            'train-load': 3,
        }

    def _get_int_bound(self, fluent_name):
        bounds = self.instance_data.get('bounds', {})
        logical_name = self.bound_aliases.get(fluent_name, fluent_name)

        explicit_values = {}
        for key in {fluent_name, logical_name}:
            if key in bounds:
                explicit_values[key] = bounds[key]

        if len(set(explicit_values.values())) > 1:
            raise ValueError(
                f"Conflicting bounds for '{fluent_name}' and its logical alias '{logical_name}': {explicit_values}"
            )

        if explicit_values:
            bound = next(iter(explicit_values.values()))
        elif logical_name in self.default_bounds:
            bound = self.default_bounds[logical_name]
        elif fluent_name in self.default_bounds:
            bound = self.default_bounds[fluent_name]
        else:
            raise ValueError(
                f"Missing integer bound for '{fluent_name}' in instance 'bounds'."
            )
        if not isinstance(bound, int) or bound < 0:
            raise ValueError(f"Bound for '{fluent_name}' must be a non-negative integer, got: {bound}")
        return bound

    def bounded_int_type(self, fluent_name):
        return IntType(0, self._get_int_bound(fluent_name))
    
    def add_social_law(self):
        """
        Add social law constraints to the CIV domain:
        1. Agent-specific available resources: Each agent gets a personal quota of resources
        2. Waitfor annotations: Movement capacity preconditions become non-blocking waits
        3. Boats must end in safe wharf positions to avoid abandoned-boat deadlocks
        """
        sl = SocialLaw()
        sl.skip_checks = True
        
        # 1. Add agent-specific available fluents for each resource at each place
        for agent in self.problem.agents:
            # For each resource, create personal_available(r, p) fluent
            sl.add_new_fluent(agent.name, 'personal-available', (('r', 'resource'), ('p', 'place')), 0)
            
            # For each action that checks available(r, p), also check personal-available(r, p)
            # And for each effect that modifies available(r, p), also modify personal-available(r, p)
            
            # Actions that consume/check resources:
            # load-boat, unload-boat, load-train, unload-train, move_laden_cart, build_coal_stack, build_sawmill, build-mine, build_ironworks,
            # build_docks, build_wharf, build_rail, build_house, build_cart, build_train, build_ship,
            # burn_coal, saw_wood, make_iron
            
            # load/unload boat and train actions
            if self.has_boats:
                sl.add_precondition_to_action(agent.name, 'load-boat', 'personal-available', ('r', 'p'), '>=', 1)
                sl.add_effect(agent.name, 'load-boat', 'personal-available', ('r', 'p'), 1, '-')
                sl.add_effect(agent.name, 'unload-boat', 'personal-available', ('r', 'p'), 1, '+')

            if self.has_trains:
                sl.add_precondition_to_action(agent.name, 'load-train', 'personal-available', ('r', 'p'), '>=', 1)
                sl.add_effect(agent.name, 'load-train', 'personal-available', ('r', 'p'), 1, '-')
                sl.add_effect(agent.name, 'unload-train', 'personal-available', ('r', 'p'), 1, '+')
            
            # move_laden_cart action
            sl.add_precondition_to_action(agent.name, 'move-laden-cart', 'personal-available', ('r', 'p1'), '>=', 1)
            sl.add_effect(agent.name, 'move-laden-cart', 'personal-available', ('r', 'p1'), 1, '-')
            sl.add_effect(agent.name, 'move-laden-cart', 'personal-available', ('r', 'p2'), 1, '+')
            
            # build_coal_stack
            sl.add_precondition_to_action(agent.name, 'build-coal-stack', 'personal-available', ('r', 'p'), '>=', 1)
            sl.add_effect(agent.name, 'build-coal-stack', 'personal-available', ('r', 'p'), 1, '-')
            
            # build_sawmill
            sl.add_precondition_to_action(agent.name, 'build-sawmill', 'personal-available', ('r', 'p'), '>=', 2)
            sl.add_effect(agent.name, 'build-sawmill', 'personal-available', ('r', 'p'), 2, '-')
            
            # build_mine
            sl.add_precondition_to_action(agent.name, 'build-mine', 'personal-available', ('r', 'p'), '>=', 2)
            sl.add_effect(agent.name, 'build-mine', 'personal-available', ('r', 'p'), 2, '-')
            
            # build_ironworks (uses r1 and r2)
            sl.add_precondition_to_action(agent.name, 'build-ironworks', 'personal-available', ('r1', 'p'), '>=', 2)
            sl.add_effect(agent.name, 'build-ironworks', 'personal-available', ('r1', 'p'), 2, '-')
            sl.add_precondition_to_action(agent.name, 'build-ironworks', 'personal-available', ('r2', 'p'), '>=', 2)
            sl.add_effect(agent.name, 'build-ironworks', 'personal-available', ('r2', 'p'), 2, '-')
            
            # build_docks (uses r1 and r2)
            sl.add_precondition_to_action(agent.name, 'build-docks', 'personal-available', ('r1', 'p'), '>=', 2)
            sl.add_effect(agent.name, 'build-docks', 'personal-available', ('r1', 'p'), 2, '-')
            sl.add_precondition_to_action(agent.name, 'build-docks', 'personal-available', ('r2', 'p'), '>=', 2)
            sl.add_effect(agent.name, 'build-docks', 'personal-available', ('r2', 'p'), 2, '-')
            
            if self.has_boats:
                sl.add_precondition_to_action(agent.name, 'build-wharf', 'personal-available', ('r1', 'p'), '>=', 2)
                sl.add_effect(agent.name, 'build-wharf', 'personal-available', ('r1', 'p'), 2, '-')
                sl.add_precondition_to_action(agent.name, 'build-wharf', 'personal-available', ('r2', 'p'), '>=', 2)
                sl.add_effect(agent.name, 'build-wharf', 'personal-available', ('r2', 'p'), 2, '-')
            
            # build_rail (uses r1 and r2)
            sl.add_precondition_to_action(agent.name, 'build-rail', 'personal-available', ('r1', 'p1'), '>=', 1)
            sl.add_effect(agent.name, 'build-rail', 'personal-available', ('r1', 'p1'), 1, '-')
            sl.add_precondition_to_action(agent.name, 'build-rail', 'personal-available', ('r2', 'p1'), '>=', 1)
            sl.add_effect(agent.name, 'build-rail', 'personal-available', ('r2', 'p1'), 1, '-')
            
            # build_house (uses r1 and r2)
            sl.add_precondition_to_action(agent.name, 'build-house', 'personal-available', ('r1', 'p'), '>=', 1)
            sl.add_effect(agent.name, 'build-house', 'personal-available', ('r1', 'p'), 1, '-')
            sl.add_precondition_to_action(agent.name, 'build-house', 'personal-available', ('r2', 'p'), '>=', 1)
            sl.add_effect(agent.name, 'build-house', 'personal-available', ('r2', 'p'), 1, '-')
            
            # build_cart
            sl.add_precondition_to_action(agent.name, 'build-cart', 'personal-available', ('r', 'p'), '>=', 1)
            sl.add_effect(agent.name, 'build-cart', 'personal-available', ('r', 'p'), 1, '-')
            
            if self.has_trains:
                sl.add_precondition_to_action(agent.name, 'build-train', 'personal-available', ('r', 'p'), '>=', 2)
                sl.add_effect(agent.name, 'build-train', 'personal-available', ('r', 'p'), 2, '-')
            
            if self.has_boats:
                sl.add_precondition_to_action(agent.name, 'build-ship', 'personal-available', ('r', 'p'), '>=', 4)
                sl.add_effect(agent.name, 'build-ship', 'personal-available', ('r', 'p'), 4, '-')
            
            # burn_coal (uses r1 and r2)
            sl.add_precondition_to_action(agent.name, 'burn-coal', 'personal-available', ('r1', 'p'), '>=', 1)
            sl.add_effect(agent.name, 'burn-coal', 'personal-available', ('r1', 'p'), 1, '-')
            sl.add_effect(agent.name, 'burn-coal', 'personal-available', ('r2', 'p'), 1, '+')
            
            # saw_wood (uses r1 and r2)
            sl.add_precondition_to_action(agent.name, 'saw-wood', 'personal-available', ('r1', 'p'), '>=', 1)
            sl.add_effect(agent.name, 'saw-wood', 'personal-available', ('r1', 'p'), 1, '-')
            sl.add_effect(agent.name, 'saw-wood', 'personal-available', ('r2', 'p'), 1, '+')
            
            # make_iron (uses r1, r2, produces r3)
            sl.add_precondition_to_action(agent.name, 'make-iron', 'personal-available', ('r1', 'p'), '>=', 1)
            sl.add_effect(agent.name, 'make-iron', 'personal-available', ('r1', 'p'), 1, '-')
            sl.add_precondition_to_action(agent.name, 'make-iron', 'personal-available', ('r2', 'p'), '>=', 2)
            sl.add_effect(agent.name, 'make-iron', 'personal-available', ('r2', 'p'), 2, '-')
            sl.add_effect(agent.name, 'make-iron', 'personal-available', ('r3', 'p'), 1, '+')
            
            # fell_timber, break_stone, mine_ore produce resources
            sl.add_effect(agent.name, 'fell-timber', 'personal-available', ('r', 'p'), 1, '+')
            sl.add_effect(agent.name, 'break-stone', 'personal-available', ('r', 'p'), 1, '+')
            sl.add_effect(agent.name, 'mine-ore', 'personal-available', ('r', 'p'), 1, '+')
            
        # 2. Add waitfor annotations for movement-capacity preconditions
        # move_train and move_ship have preconditions on train/boat capacity
        # Convert to waitfor - wait until capacity is positive
        for agent in self.problem.agents:
            if self.has_trains:
                sl.add_waitfor_annotation(agent.name, 'move-train', 'train-capacity', ('t', 'p2'), '>=', 1)
            if self.has_boats:
                sl.add_waitfor_annotation(agent.name, 'move-ship', 'boat-capacity', ('p2',), '>=', 1)
                sl.add_waitfor_annotation(agent.name, 'move-ship-to-wharf', 'boat-capacity', ('p2',), '>=', 1)

        # 3. Boats must finish in wharf-safe states.
        if self.has_boats:
            for agent in self.problem.agents:
                for boat_obj in self.problem.objects(self.problem.user_type('boat')):
                    sl.add_agent_goal(agent.name, 'boat-at-wharf', (boat_obj.name,))
        
        self.problem = sl.compile(self.problem).problem
        return self.problem
    
    def generate_problem(self, file_name, sl=False):
        self.problem = MultiAgentProblemWithWaitfor('civ_' + file_name.replace('.json', ''))
        self.load_instance_data(file_name)
        self.has_boats = len(self.instance_data.get('boats', [])) > 0
        self.has_trains = len(self.instance_data.get('trains', [])) > 0
        
        # Define types
        place = UserType('place')
        boat = UserType('boat')
        train = UserType('train')
        resource_type = UserType('resource')
        
        # Load objects
        self.load_objects(['places', 'boats', 'trains', 'resources'], [place, boat, train, resource_type])
        
        # ==================== GLOBAL/ENVIRONMENT FLUENTS ====================
        # Connection and location properties
        connected_by_land = Fluent('connected-by-land', BoolType(), p1=place, p2=place)
        connected_by_rail = Fluent('connected-by-rail', BoolType(), p1=place, p2=place)
        connected_by_sea = Fluent('connected-by-sea', BoolType(), p1=place, p2=place)
        woodland = Fluent('woodland', BoolType(), p=place)
        mountain = Fluent('mountain', BoolType(), p=place)
        metalliferous = Fluent('metalliferous', BoolType(), p=place)
        by_coast = Fluent('by-coast', BoolType(), p=place)

        is_timber = Fluent('is_timber', BoolType(), r=resource_type)
        is_stone = Fluent('is_stone', BoolType(), r=resource_type)
        is_ore = Fluent('is_ore', BoolType(), r=resource_type)
        is_coal = Fluent('is_coal', BoolType(), r=resource_type)
        is_wood = Fluent('is_wood', BoolType(), r=resource_type)
        is_iron = Fluent('is_iron', BoolType(), r=resource_type)

        # Building properties (global)
        has_coal_stack = Fluent('has-coal-stack', BoolType(), p=place)
        has_quarry = Fluent('has-quarry', BoolType(), p=place)
        has_mine = Fluent('has-mine', BoolType(), p=place)
        has_sawmill = Fluent('has-sawmill', BoolType(), p=place)
        has_ironworks = Fluent('has-ironworks', BoolType(), p=place)
        has_docks = Fluent('has-docks', BoolType(), p=place)
        has_wharf = Fluent('has-wharf', BoolType(), p=place)
        
        # Vehicle properties and tracking
        boat_at = Fluent('boat-at', BoolType(), b=boat, p=place) if self.has_boats else None
        train_at = Fluent('train-at', BoolType(), t=train, p=place) if self.has_trains else None
        boat_potential = Fluent('boat-potential', BoolType(), b=boat) if self.has_boats else None
        train_potential = Fluent('train-potential', BoolType(), t=train) if self.has_trains else None
        boat_at_wharf = Fluent('boat-at-wharf', BoolType(), b=boat) if self.has_boats else None
        
        # Global counters
        pollution = Fluent('pollution', self.bounded_int_type('pollution'))
        labour = Fluent('labour', self.bounded_int_type('labour'))
        resource_use = Fluent('resource-use', self.bounded_int_type('resource-use'))
        housing = Fluent('housing', self.bounded_int_type('housing'), p=place)
        
        # SHARED RESOURCES at locations (creates agent contention!)
        available = Fluent('available', self.bounded_int_type('available'), r=resource_type, p=place)

        # Capacity management
        boat_capacity = Fluent('boat-capacity', self.bounded_int_type('boat-capacity'), p=place) if self.has_boats else None
        train_capacity = Fluent('train-capacity', self.bounded_int_type('train-capacity'), t=train, p=place) if self.has_trains else None
        
        # Add all global fluents to environment
        self.problem.ma_environment.add_fluent(connected_by_land, default_initial_value=False)
        self.problem.ma_environment.add_fluent(connected_by_rail, default_initial_value=False)
        self.problem.ma_environment.add_fluent(connected_by_sea, default_initial_value=False)
        self.problem.ma_environment.add_fluent(woodland, default_initial_value=False)
        self.problem.ma_environment.add_fluent(mountain, default_initial_value=False)
        self.problem.ma_environment.add_fluent(metalliferous, default_initial_value=False)
        self.problem.ma_environment.add_fluent(by_coast, default_initial_value=False)

        self.problem.ma_environment.add_fluent(is_timber, default_initial_value=False)
        self.problem.ma_environment.add_fluent(is_stone, default_initial_value=False)
        self.problem.ma_environment.add_fluent(is_ore, default_initial_value=False)
        self.problem.ma_environment.add_fluent(is_coal, default_initial_value=False)
        self.problem.ma_environment.add_fluent(is_wood, default_initial_value=False)
        self.problem.ma_environment.add_fluent(is_iron, default_initial_value=False)

        self.problem.ma_environment.add_fluent(has_quarry, default_initial_value=False)
        self.problem.ma_environment.add_fluent(has_mine, default_initial_value=False)
        self.problem.ma_environment.add_fluent(has_ironworks, default_initial_value=False)
        self.problem.ma_environment.add_fluent(has_docks, default_initial_value=False)
        self.problem.ma_environment.add_fluent(has_wharf, default_initial_value=False)
        
        if self.has_boats:
            self.problem.ma_environment.add_fluent(boat_at, default_initial_value=False)
            self.problem.ma_environment.add_fluent(boat_potential, default_initial_value=True)
            self.problem.ma_environment.add_fluent(boat_at_wharf, default_initial_value=True)
        if self.has_trains:
            self.problem.ma_environment.add_fluent(train_at, default_initial_value=False)
            self.problem.ma_environment.add_fluent(train_potential, default_initial_value=True)
        
        self.problem.ma_environment.add_fluent(pollution, default_initial_value=0)
        self.problem.ma_environment.add_fluent(resource_use, default_initial_value=0)
        self.problem.ma_environment.add_fluent(housing, default_initial_value=0)

        self.problem.ma_environment.add_fluent(available, default_initial_value=0)  # SHARED resources at locations
        if self.has_boats:
            self.problem.ma_environment.add_fluent(boat_capacity, default_initial_value=0)
        if self.has_trains:
            self.problem.ma_environment.add_fluent(train_capacity, default_initial_value=0)
        
        # ==================== AGENT-SPECIFIC FLUENTS ====================
        # Labour (citizens controlled by each agent)
        labour = Fluent('labour', self.bounded_int_type('labour'))
        has_cabin = Fluent('has-cabin', BoolType(), p=place)

        # Ownership
        owns_boat = Fluent('owns_boat', BoolType(), b=boat) if self.has_boats else None
        owns_train = Fluent('owns_train', BoolType(), t=train) if self.has_trains else None
        assigned_boat = Fluent('assigned-boat', BoolType(), b=boat) if self.has_boats else None
        assigned_train = Fluent('assigned-train', BoolType(), t=train) if self.has_trains else None

        # Carts at locations for each agent
        carts_at = Fluent('carts-at', self.bounded_int_type('carts-at'), p=place)

        # Vehicle cargo and capacity bookkeeping is global: it is a property of the object,
        # not of a particular agent's local state.
        resources_in_boat = Fluent('resources-in-boat', self.bounded_int_type('resources-in-boat'), b=boat, r=resource_type) if self.has_boats else None
        resources_in_train = Fluent('resources-in-train', self.bounded_int_type('resources-in-train'), t=train, r=resource_type) if self.has_trains else None
        boat_space_in = Fluent('boat-space-in', self.bounded_int_type('boat-space-in'), b=boat) if self.has_boats else None
        train_space_in = Fluent('train-space-in', self.bounded_int_type('train-space-in'), t=train) if self.has_trains else None
        if self.has_boats:
            self.problem.ma_environment.add_fluent(resources_in_boat, default_initial_value=0)
            self.problem.ma_environment.add_fluent(boat_space_in, default_initial_value=0)
        if self.has_trains:
            self.problem.ma_environment.add_fluent(resources_in_train, default_initial_value=0)
            self.problem.ma_environment.add_fluent(train_space_in, default_initial_value=0)
        
        # ==================== ACTIONS ====================
        def inc(action, fluent_exp, amount):
            action.add_increase_effect(fluent_exp, amount)

        def dec(action, fluent_exp, amount):
            action.add_decrease_effect(fluent_exp, amount)
        
        # A.1: Load and unload resources (agent-specific)
        load_boat = unload_boat = None
        if self.has_boats:
            load_boat = InstantaneousAction('load-boat', b=boat, p=place, r=resource_type)
            load_boat_b = load_boat.parameter('b')
            load_boat_p = load_boat.parameter('p')
            load_boat_r = load_boat.parameter('r')
            load_boat.add_precondition(boat_at(load_boat_b, load_boat_p))
            load_boat.add_precondition(GE(available(load_boat_r, load_boat_p), 1))
            load_boat.add_precondition(GE(boat_space_in(load_boat_b), 1))
            dec(load_boat, boat_space_in(load_boat_b), 1)
            dec(load_boat, available(load_boat_r, load_boat_p), 1)
            inc(load_boat, resources_in_boat(load_boat_b, load_boat_r), 1)
            inc(load_boat, labour, 1)

            unload_boat = InstantaneousAction('unload-boat', b=boat, p=place, r=resource_type)
            unload_boat_b = unload_boat.parameter('b')
            unload_boat_p = unload_boat.parameter('p')
            unload_boat_r = unload_boat.parameter('r')
            unload_boat.add_precondition(boat_at(unload_boat_b, unload_boat_p))
            inc(unload_boat, boat_space_in(unload_boat_b), 1)
            inc(unload_boat, available(unload_boat_r, unload_boat_p), 1)
            dec(unload_boat, resources_in_boat(unload_boat_b, unload_boat_r), 1)
            inc(unload_boat, labour, 1)

        load_train = unload_train = None
        if self.has_trains:
            load_train = InstantaneousAction('load-train', t=train, p=place, r=resource_type)
            load_train_t = load_train.parameter('t')
            load_train_p = load_train.parameter('p')
            load_train_r = load_train.parameter('r')
            load_train.add_precondition(train_at(load_train_t, load_train_p))
            load_train.add_precondition(GE(available(load_train_r, load_train_p), 1))
            load_train.add_precondition(GE(train_space_in(load_train_t), 1))
            dec(load_train, train_space_in(load_train_t), 1)
            dec(load_train, available(load_train_r, load_train_p), 1)
            inc(load_train, resources_in_train(load_train_t, load_train_r), 1)
            inc(load_train, labour, 1)

            unload_train = InstantaneousAction('unload-train', t=train, p=place, r=resource_type)
            unload_train_t = unload_train.parameter('t')
            unload_train_p = unload_train.parameter('p')
            unload_train_r = unload_train.parameter('r')
            unload_train.add_precondition(train_at(unload_train_t, unload_train_p))
            inc(unload_train, train_space_in(unload_train_t), 1)
            inc(unload_train, available(unload_train_r, unload_train_p), 1)
            dec(unload_train, resources_in_train(unload_train_t, unload_train_r), 1)
            inc(unload_train, labour, 1)

        # A.2: Move carts
        move_empty_cart = InstantaneousAction('move-empty-cart', p1=place, p2=place)
        move_empty_cart_p1 = move_empty_cart.parameter('p1')
        move_empty_cart_p2 = move_empty_cart.parameter('p2')
        move_empty_cart.add_precondition(connected_by_land(move_empty_cart_p1, move_empty_cart_p2))
        move_empty_cart.add_precondition(GE(carts_at(move_empty_cart_p1), 1))
        dec(move_empty_cart, carts_at(move_empty_cart_p1), 1)
        inc(move_empty_cart, carts_at(move_empty_cart_p2), 1)
        inc(move_empty_cart, labour, 2)

        move_laden_cart = InstantaneousAction('move-laden-cart', p1=place, p2=place, r=resource_type)
        move_laden_cart_p1 = move_laden_cart.parameter('p1')
        move_laden_cart_p2 = move_laden_cart.parameter('p2')
        move_laden_cart_r = move_laden_cart.parameter('r')
        move_laden_cart.add_precondition(connected_by_land(move_laden_cart_p1, move_laden_cart_p2))
        move_laden_cart.add_precondition(GE(carts_at(move_laden_cart_p1), 1))
        move_laden_cart.add_precondition(GE(available(move_laden_cart_r, move_laden_cart_p1), 1))
        dec(move_laden_cart, carts_at(move_laden_cart_p1), 1)
        inc(move_laden_cart, carts_at(move_laden_cart_p2), 1)
        inc(move_laden_cart, labour, 2)
        
        # A.2: Move train
        move_train = None
        if self.has_trains:
            move_train = InstantaneousAction('move-train', t=train, p1=place, p2=place)
            move_train_t = move_train.parameter('t')
            move_train_p1 = move_train.parameter('p1')
            move_train_p2 = move_train.parameter('p2')
            move_train.add_precondition(owns_train(move_train_t))
            move_train.add_precondition(connected_by_rail(move_train_p1, move_train_p2))
            move_train.add_precondition(train_at(move_train_t, move_train_p1))
            move_train.add_precondition(GE(train_capacity(move_train_t, move_train_p2), 1))
            move_train.add_effect(train_at(move_train_t, move_train_p1), False)
            move_train.add_effect(train_at(move_train_t, move_train_p2), True)
            inc(move_train, train_capacity(move_train_t, move_train_p1), 1)
            dec(move_train, train_capacity(move_train_t, move_train_p2), 1)
            inc(move_train, pollution, 1)
        
        # A.2: Move ship
        move_ship = move_ship_to_wharf = None
        if self.has_boats:
            move_ship = InstantaneousAction('move-ship', b=boat, p1=place, p2=place)
            move_ship_b = move_ship.parameter('b')
            move_ship_p1 = move_ship.parameter('p1')
            move_ship_p2 = move_ship.parameter('p2')
            move_ship.add_precondition(owns_boat(move_ship_b))
            move_ship.add_precondition(connected_by_sea(move_ship_p1, move_ship_p2))
            move_ship.add_precondition(boat_at(move_ship_b, move_ship_p1))
            move_ship.add_precondition(Not(has_wharf(move_ship_p2)))
            move_ship.add_precondition(GE(boat_capacity(move_ship_p2), 1))
            move_ship.add_effect(boat_at(move_ship_b, move_ship_p1), False)
            move_ship.add_effect(boat_at(move_ship_b, move_ship_p2), True)
            inc(move_ship, boat_capacity(move_ship_p1), 1)
            dec(move_ship, boat_capacity(move_ship_p2), 1)
            move_ship.add_effect(boat_at_wharf(move_ship_b), False)
            inc(move_ship, pollution, 2)

            move_ship_to_wharf = InstantaneousAction('move-ship-to-wharf', b=boat, p1=place, p2=place)
            move_ship_to_wharf_b = move_ship_to_wharf.parameter('b')
            move_ship_to_wharf_p1 = move_ship_to_wharf.parameter('p1')
            move_ship_to_wharf_p2 = move_ship_to_wharf.parameter('p2')
            move_ship_to_wharf.add_precondition(owns_boat(move_ship_to_wharf_b))
            move_ship_to_wharf.add_precondition(connected_by_sea(move_ship_to_wharf_p1, move_ship_to_wharf_p2))
            move_ship_to_wharf.add_precondition(boat_at(move_ship_to_wharf_b, move_ship_to_wharf_p1))
            move_ship_to_wharf.add_precondition(has_wharf(move_ship_to_wharf_p2))
            move_ship_to_wharf.add_precondition(GE(boat_capacity(move_ship_to_wharf_p2), 1))
            move_ship_to_wharf.add_effect(boat_at(move_ship_to_wharf_b, move_ship_to_wharf_p1), False)
            move_ship_to_wharf.add_effect(boat_at(move_ship_to_wharf_b, move_ship_to_wharf_p2), True)
            inc(move_ship_to_wharf, boat_capacity(move_ship_to_wharf_p1), 1)
            dec(move_ship_to_wharf, boat_capacity(move_ship_to_wharf_p2), 1)
            move_ship_to_wharf.add_effect(boat_at_wharf(move_ship_to_wharf_b), True)
            inc(move_ship_to_wharf, pollution, 2)
        
        # B.1: Build structures
        build_cabin = InstantaneousAction('build-cabin', p=place)
        build_cabin_p = build_cabin.parameter('p')
        build_cabin.add_precondition(woodland(build_cabin_p))
        build_cabin.add_precondition(Not(has_cabin(build_cabin_p)))
        build_cabin.add_effect(has_cabin(build_cabin_p), True)
        inc(build_cabin, labour, 1)
        
        build_quarry = InstantaneousAction('build-quarry', p=place)
        build_quarry_p = build_quarry.parameter('p')
        build_quarry.add_precondition(mountain(build_quarry_p))
        build_quarry.add_precondition(Not(has_quarry(build_quarry_p)))
        build_quarry.add_effect(has_quarry(build_quarry_p), True)
        inc(build_quarry, labour, 2)
        
        build_coal_stack = InstantaneousAction('build-coal-stack', p=place, r=resource_type)
        build_coal_stack_p = build_coal_stack.parameter('p')
        build_coal_stack_r = build_coal_stack.parameter('r')  # timber
        build_coal_stack.add_precondition(is_timber(build_coal_stack_r))
        build_coal_stack.add_precondition(GE(available(build_coal_stack_r, build_coal_stack_p), 1))
        build_coal_stack.add_precondition(GE(available(build_coal_stack_r, build_coal_stack_p), 1))
        build_coal_stack.add_precondition(GE(available(build_coal_stack_r, build_coal_stack_p), 1))
        build_coal_stack.add_precondition(Not(has_coal_stack(build_coal_stack_p)))
        build_coal_stack.add_effect(has_coal_stack(build_coal_stack_p), True)
        dec(build_coal_stack, available(build_coal_stack_r, build_coal_stack_p), 1)
        inc(build_coal_stack, labour, 2)
        
        build_sawmill = InstantaneousAction('build-sawmill', p=place, r=resource_type)
        build_sawmill_p = build_sawmill.parameter('p')
        build_sawmill_r = build_sawmill.parameter('r')  # timber
        build_sawmill.add_precondition(is_timber(build_sawmill_r))
        build_sawmill.add_precondition(GE(available(build_sawmill_r, build_sawmill_p), 2))
        build_sawmill.add_precondition(Not(has_sawmill(build_sawmill_p)))
        build_sawmill.add_effect(has_sawmill(build_sawmill_p), True)
        dec(build_sawmill, available(build_sawmill_r, build_sawmill_p), 2)
        inc(build_sawmill, labour, 2)
        
        build_mine = InstantaneousAction('build-mine', p=place, r=resource_type)
        build_mine_p = build_mine.parameter('p')
        build_mine_r = build_mine.parameter('r')  # wood
        build_mine.add_precondition(is_wood(build_mine_r))
        build_mine.add_precondition(metalliferous(build_mine_p))
        build_mine.add_precondition(GE(available(build_mine_r, build_mine_p), 2))
        build_mine.add_precondition(Not(has_mine(build_mine_p)))
        build_mine.add_effect(has_mine(build_mine_p), True)
        dec(build_mine, available(build_mine_r, build_mine_p), 2)
        inc(build_mine, labour, 3)
        
        build_ironworks = InstantaneousAction('build-ironworks', p=place, r1=resource_type, r2=resource_type)
        build_ironworks_p = build_ironworks.parameter('p')
        build_ironworks_r1 = build_ironworks.parameter('r1')  # stone
        build_ironworks_r2 = build_ironworks.parameter('r2')  # wood
        build_ironworks.add_precondition(is_stone(build_ironworks_r1))
        build_ironworks.add_precondition(is_wood(build_ironworks_r2))
        build_ironworks.add_precondition(GE(available(build_ironworks_r1, build_ironworks_p), 2))
        build_ironworks.add_precondition(GE(available(build_ironworks_r2, build_ironworks_p), 2))
        build_ironworks.add_precondition(Not(has_ironworks(build_ironworks_p)))
        build_ironworks.add_effect(has_ironworks(build_ironworks_p), True)
        dec(build_ironworks, available(build_ironworks_r1, build_ironworks_p), 2)
        dec(build_ironworks, available(build_ironworks_r2, build_ironworks_p), 2)
        inc(build_ironworks, labour, 3)
        
        build_docks = InstantaneousAction('build-docks', p=place, r1=resource_type, r2=resource_type)
        build_docks_p = build_docks.parameter('p')
        build_docks_r1 = build_docks.parameter('r1')  # stone
        build_docks_r2 = build_docks.parameter('r2')  # wood
        build_docks.add_precondition(is_stone(build_docks_r1))
        build_docks.add_precondition(is_wood(build_docks_r2))
        build_docks.add_precondition(by_coast(build_docks_p))
        build_docks.add_precondition(GE(available(build_docks_r1, build_docks_p), 2))
        build_docks.add_precondition(GE(available(build_docks_r2, build_docks_p), 2))
        build_docks.add_precondition(Not(has_docks(build_docks_p)))
        build_docks.add_effect(has_docks(build_docks_p), True)
        dec(build_docks, available(build_docks_r1, build_docks_p), 2)
        dec(build_docks, available(build_docks_r2, build_docks_p), 2)
        inc(build_docks, labour, 2)
        
        build_wharf = None
        if self.has_boats:
            build_wharf = InstantaneousAction('build-wharf', p=place, r1=resource_type, r2=resource_type)
            build_wharf_p = build_wharf.parameter('p')
            build_wharf_r1 = build_wharf.parameter('r1')  # stone
            build_wharf_r2 = build_wharf.parameter('r2')  # iron
            build_wharf.add_precondition(is_stone(build_wharf_r1))
            build_wharf.add_precondition(is_iron(build_wharf_r2))
            build_wharf.add_precondition(has_docks(build_wharf_p))
            build_wharf.add_precondition(Not(has_wharf(build_wharf_p)))
            build_wharf.add_precondition(GE(available(build_wharf_r1, build_wharf_p), 2))
            build_wharf.add_precondition(GE(available(build_wharf_r2, build_wharf_p), 2))
            build_wharf.add_effect(has_wharf(build_wharf_p), True)
            inc(build_wharf, boat_capacity(build_wharf_p), 1)
            dec(build_wharf, available(build_wharf_r1, build_wharf_p), 2)
            dec(build_wharf, available(build_wharf_r2, build_wharf_p), 2)
            inc(build_wharf, labour, 2)
        
        build_rail = InstantaneousAction('build-rail', p1=place, p2=place, r1=resource_type, r2=resource_type)
        build_rail_p1 = build_rail.parameter('p1')
        build_rail_p2 = build_rail.parameter('p2')
        build_rail_r1 = build_rail.parameter('r1')  # wood
        build_rail_r2 = build_rail.parameter('r2')  # iron
        build_rail.add_precondition(is_wood(build_rail_r1))
        build_rail.add_precondition(is_iron(build_rail_r2))
        build_rail.add_precondition(connected_by_land(build_rail_p1, build_rail_p2))
        build_rail.add_precondition(GE(available(build_rail_r1, build_rail_p1), 1))
        build_rail.add_precondition(GE(available(build_rail_r2, build_rail_p1), 1))
        build_rail.add_effect(connected_by_rail(build_rail_p1, build_rail_p2), True)
        dec(build_rail, available(build_rail_r1, build_rail_p1), 1)
        dec(build_rail, available(build_rail_r2, build_rail_p1), 1)
        inc(build_rail, labour, 2)
        
        build_house = InstantaneousAction('build-house', p=place, r1=resource_type, r2=resource_type)
        build_house_p = build_house.parameter('p')
        build_house_r1 = build_house.parameter('r1')  # wood
        build_house_r2 = build_house.parameter('r2')  # stone
        build_house.add_precondition(is_wood(build_house_r1))
        build_house.add_precondition(is_stone(build_house_r2))
        build_house.add_precondition(GE(available(build_house_r1, build_house_p), 1))
        build_house.add_precondition(GE(available(build_house_r2, build_house_p), 1))
        inc(build_house, housing(build_house_p), 1)
        dec(build_house, available(build_house_r1, build_house_p), 1)
        dec(build_house, available(build_house_r2, build_house_p), 1)

        # B.2: Build vehicles and carts
        build_cart = InstantaneousAction('build-cart', p=place, r=resource_type)
        build_cart_p = build_cart.parameter('p')
        build_cart_r = build_cart.parameter('r')  # timber
        build_cart.add_precondition(is_timber(build_cart_r))
        build_cart.add_precondition(GE(available(build_cart_r, build_cart_p), 1))
        inc(build_cart, carts_at(build_cart_p), 1)
        dec(build_cart, available(build_cart_r, build_cart_p), 1)
        inc(build_cart, labour, 1)
        
        build_train = None
        if self.has_trains:
            build_train = InstantaneousAction('build-train', p=place, t=train, r=resource_type)
            build_train_p = build_train.parameter('p')
            build_train_t = build_train.parameter('t')
            build_train_r = build_train.parameter('r')  # iron
            build_train.add_precondition(is_iron(build_train_r))
            build_train.add_precondition(assigned_train(build_train_t))
            build_train.add_precondition(train_potential(build_train_t))
            build_train.add_precondition(GE(available(build_train_r, build_train_p), 2))
            build_train.add_effect(owns_train(build_train_t), True)
            build_train.add_effect(train_at(build_train_t, build_train_p), True)
            build_train.add_effect(train_potential(build_train_t), False)
            dec(build_train, train_capacity(build_train_t, build_train_p), 1)
            dec(build_train, available(build_train_r, build_train_p), 2)
            inc(build_train, labour, 2)
        
        build_ship = None
        if self.has_boats:
            build_ship = InstantaneousAction('build-ship', p=place, b=boat, r=resource_type)
            build_ship_p = build_ship.parameter('p')
            build_ship_b = build_ship.parameter('b')
            build_ship_r = build_ship.parameter('r')  # iron
            build_ship.add_precondition(is_iron(build_ship_r))
            build_ship.add_precondition(assigned_boat(build_ship_b))
            build_ship.add_precondition(boat_potential(build_ship_b))
            build_ship.add_precondition(has_wharf(build_ship_p))
            build_ship.add_precondition(GE(available(build_ship_r, build_ship_p), 4))
            build_ship.add_precondition(GE(boat_capacity(build_ship_p), 1))
            build_ship.add_effect(owns_boat(build_ship_b), True)
            build_ship.add_effect(boat_at(build_ship_b, build_ship_p), True)
            build_ship.add_effect(boat_potential(build_ship_b), False)
            build_ship.add_effect(boat_at_wharf(build_ship_b), True)
            dec(build_ship, boat_capacity(build_ship_p), 1)
            dec(build_ship, available(build_ship_r, build_ship_p), 4)
            inc(build_ship, labour, 3)
        
        # C.1: Obtain raw resources
        fell_timber = InstantaneousAction('fell-timber', p=place, r=resource_type)
        fell_timber_p = fell_timber.parameter('p')
        fell_timber_r = fell_timber.parameter('r')  # timber
        fell_timber.add_precondition(is_timber(fell_timber_r))
        fell_timber.add_precondition(has_cabin(fell_timber_p))
        inc(fell_timber, available(fell_timber_r, fell_timber_p), 1)
        inc(fell_timber, labour, 1)
        
        break_stone = InstantaneousAction('break-stone', p=place, r=resource_type)
        break_stone_p = break_stone.parameter('p')
        break_stone_r = break_stone.parameter('r')  # stone
        break_stone.add_precondition(is_stone(break_stone_r))
        break_stone.add_precondition(has_quarry(break_stone_p))
        inc(break_stone, available(break_stone_r, break_stone_p), 1)
        inc(break_stone, labour, 1)
        inc(break_stone, resource_use, 1)
        
        mine_ore = InstantaneousAction('mine-ore', p=place, r=resource_type)
        mine_ore_p = mine_ore.parameter('p')
        mine_ore_r = mine_ore.parameter('r')  # ore
        mine_ore.add_precondition(is_ore(mine_ore_r))
        mine_ore.add_precondition(has_mine(mine_ore_p))
        inc(mine_ore, available(mine_ore_r, mine_ore_p), 1)
        inc(mine_ore, resource_use, 2)
        
        # C.2: Refine resources
        burn_coal = InstantaneousAction('burn-coal', p=place, r1=resource_type, r2=resource_type)
        burn_coal_p = burn_coal.parameter('p')
        burn_coal_r1 = burn_coal.parameter('r1')  # timber
        burn_coal_r2 = burn_coal.parameter('r2')  # coal
        burn_coal.add_precondition(is_timber(burn_coal_r1))
        burn_coal.add_precondition(is_coal(burn_coal_r2))
        burn_coal.add_precondition(has_coal_stack(burn_coal_p))
        burn_coal.add_precondition(GE(available(burn_coal_r1, burn_coal_p), 1))
        dec(burn_coal, available(burn_coal_r1, burn_coal_p), 1)
        inc(burn_coal, available(burn_coal_r2, burn_coal_p), 1)
        inc(burn_coal, pollution, 1)
        
        saw_wood = InstantaneousAction('saw-wood', p=place, r1=resource_type, r2=resource_type)
        saw_wood_p = saw_wood.parameter('p')
        saw_wood_r1 = saw_wood.parameter('r1')  # timber
        saw_wood_r2 = saw_wood.parameter('r2')  # wood
        saw_wood.add_precondition(is_timber(saw_wood_r1))
        saw_wood.add_precondition(is_wood(saw_wood_r2))
        saw_wood.add_precondition(has_sawmill(saw_wood_p))
        saw_wood.add_precondition(GE(available(saw_wood_r1, saw_wood_p), 1))
        dec(saw_wood, available(saw_wood_r1, saw_wood_p), 1)
        inc(saw_wood, available(saw_wood_r2, saw_wood_p), 1)
        
        make_iron = InstantaneousAction('make-iron', p=place, r1=resource_type, r2=resource_type, r3=resource_type)
        make_iron_p = make_iron.parameter('p')
        make_iron_r1 = make_iron.parameter('r1')  # ore
        make_iron_r2 = make_iron.parameter('r2')  # coal
        make_iron_r3 = make_iron.parameter('r3')  # iron
        make_iron.add_precondition(is_ore(make_iron_r1))
        make_iron.add_precondition(is_coal(make_iron_r2))
        make_iron.add_precondition(is_iron(make_iron_r3))
        make_iron.add_precondition(has_ironworks(make_iron_p))
        make_iron.add_precondition(GE(available(make_iron_r1, make_iron_p), 1))
        make_iron.add_precondition(GE(available(make_iron_r2, make_iron_p), 2))
        dec(make_iron, available(make_iron_r1, make_iron_p), 1)
        dec(make_iron, available(make_iron_r2, make_iron_p), 2)
        inc(make_iron, available(make_iron_r3, make_iron_p), 1)
        inc(make_iron, pollution, 2)
        
        # D: Dismantle vehicles
        dismantle_train = None
        if self.has_trains:
            dismantle_train = InstantaneousAction('dismantle-train', t=train, p=place)
            dismantle_train_t = dismantle_train.parameter('t')
            dismantle_train_p = dismantle_train.parameter('p')
            dismantle_train.add_precondition(train_at(dismantle_train_t, dismantle_train_p))
            dismantle_train.add_effect(train_potential(dismantle_train_t), True)
            dismantle_train.add_effect(owns_train(dismantle_train_t), False)
            dismantle_train.add_effect(train_at(dismantle_train_t, dismantle_train_p), False)
            inc(dismantle_train, train_capacity(dismantle_train_t, dismantle_train_p), 1)

        dismantle_boat = None
        if self.has_boats:
            dismantle_boat = InstantaneousAction('dismantle-boat', b=boat, p=place)
            dismantle_boat_b = dismantle_boat.parameter('b')
            dismantle_boat_p = dismantle_boat.parameter('p')
            dismantle_boat.add_precondition(boat_at(dismantle_boat_b, dismantle_boat_p))
            dismantle_boat.add_effect(boat_potential(dismantle_boat_b), True)
            dismantle_boat.add_effect(owns_boat(dismantle_boat_b), False)
            dismantle_boat.add_effect(boat_at(dismantle_boat_b, dismantle_boat_p), False)
            dismantle_boat.add_effect(boat_at_wharf(dismantle_boat_b), True)
            inc(dismantle_boat, boat_capacity(dismantle_boat_p), 1)
        
        # ==================== CREATE AGENTS ====================
        self.load_agents()
        
        for agent in self.problem.agents:
            # Add agent-specific fluents
            agent.add_fluent(labour, default_initial_value=0)
            agent.add_fluent(has_cabin, default_initial_value=False)
            agent.add_fluent(has_coal_stack, default_initial_value=False)
            agent.add_fluent(has_sawmill, default_initial_value=False)
            agent.add_fluent(carts_at, default_initial_value=0)
            if self.has_boats:
                agent.add_fluent(owns_boat, default_initial_value=False)
                agent.add_fluent(assigned_boat, default_initial_value=False)
            if self.has_trains:
                agent.add_fluent(owns_train, default_initial_value=False)
                agent.add_fluent(assigned_train, default_initial_value=False)

            # Add all actions to each agent
            if self.has_boats:
                agent.add_action(load_boat)
                agent.add_action(unload_boat)
            if self.has_trains:
                agent.add_action(load_train)
                agent.add_action(unload_train)
            agent.add_action(move_empty_cart)
            agent.add_action(move_laden_cart)
            if self.has_trains:
                agent.add_action(move_train)
            if self.has_boats:
                agent.add_action(move_ship)
                agent.add_action(move_ship_to_wharf)
            agent.add_action(build_cabin)
            agent.add_action(build_quarry)
            agent.add_action(build_coal_stack)
            agent.add_action(build_sawmill)
            agent.add_action(build_mine)
            agent.add_action(build_ironworks)
            agent.add_action(build_docks)
            if self.has_boats:
                agent.add_action(build_wharf)
            agent.add_action(build_rail)
            agent.add_action(build_house)
            agent.add_action(build_cart)
            if self.has_trains:
                agent.add_action(build_train)
            if self.has_boats:
                agent.add_action(build_ship)
            agent.add_action(fell_timber)
            agent.add_action(break_stone)
            agent.add_action(mine_ore)
            agent.add_action(burn_coal)
            agent.add_action(saw_wood)
            agent.add_action(make_iron)
            if self.has_trains:
                agent.add_action(dismantle_train)
            if self.has_boats:
                agent.add_action(dismantle_boat)
        
        self.set_init_values()
        self.set_goals()
        
        if sl:
            self.add_social_law()
        
        return self.problem
