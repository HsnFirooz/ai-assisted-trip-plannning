module DaytripDesignAssistance

using Distributions
using Random
using POMDPs
using POMDPModelTools
using POMDPSimulators
using POMDPPolicies
using MCTS
using JuMP
using GLPK
using LinearAlgebra
import Base.string

export POICategory, FINEARTS, POPCULTURE, SIGHTSEEING, HISTORICHOUSE, SHOPPING, HOME, POI, Daytrip, distance, travel_dist, travel_time, duration, cost, trip_value, optimize_trip, optimize_trip_cw
export DaytripEditType, DaytripEdit
export featurize
export HumanDaytripScheduleMDP, act, likelihood, get_policy, utility
export MCTSUserModel, LCUserModel, BoltzmannUserModel, MCTSUserModelSpecification, LCUserModelSpecification, BoltzmannUserModelSpecification, instantiate
export DaytripScheduleAssistanceMDP, RecommendOptimal


################################################################################
# DAYTRIP AND UTILITY DEFINITION
################################################################################

const global MAX_DAY_LENGTH = 60.0*12.0

# categories. Home should always be last so that you can sample with support 1...5
@enum POICategory FINEARTS=1 POPCULTURE=2 SIGHTSEEING=3 HISTORICHOUSE=4 SHOPPING=5 HOME=6

struct POI
    coord_x::Float64 # x coordinate in km from center
    coord_y::Float64 # y coordinate in km from center
    category::POICategory # Category of the POI
    visit_time::Float64 # visit time in minutes
    cost::Float64 # cost in eurodollars
    
    POI(coord_x::Float64, coord_y::Float64, category::POICategory, visit_time::Float64 = 0.0, cost::Float64 = 0.0) = new(coord_x, coord_y, category, visit_time, cost)
end

"""
Daytrips are represented by an array of POIs. POIs are visited in the order in which they appear in the array.
"""
Daytrip = Array{POI,1}

"""
measures the distance between POIs x and y
"""
distance(x::POI, y::POI) = sqrt((x.coord_x - y.coord_x)^2 + (x.coord_y - y.coord_y)^2)

"""
measures the included angle between AB and BC
"""
function included_angle(A::POI, B::POI, C::POI)
    BA = [A.coord_x - B.coord_x, A.coord_y - B.coord_y]
    BC = [C.coord_x - B.coord_x, C.coord_y - B.coord_y]
    acos(min(max(sum(BA .* BC) / (sqrt(sum(BA.^2)) * sqrt(sum(BC.^2))), -1.0), 1.0))
end

"""
Calculates the travel distance for a trip (in km)

parameters:
    trip::Daytrip:
        The daytrip
"""
travel_dist(trip::Daytrip)::Float64 = sum(Array{Float64,1}([distance(trip[i], trip[i%length(trip)+1]) for i in 1:length(trip)]))

"""
Calculates how long it will take to travel between all POIs in a trip (in minutes)

parameters:
    trip::Daytrip:
        The daytrip
    movement_speed::Float64:
        Speed at which distance is covered in km/h.
        default: 5.0
"""
travel_time(trip::Daytrip; movement_speed::Float64 = 5.0)::Float64 = travel_dist(trip) / movement_speed * 60.0

"""
Calculates how long it will take to visit all POIs in a trip (in minutes)

parameters:
    trip::Daytrip:
        The daytrip
    movement_speed::Float64:
        Speed at which distance is covered in km/h.
        default: 5.0
"""
function duration(trip::Daytrip; movement_speed::Float64 = 5.0)::Float64
    if length(trip) == 0
        return 0.0
    end
    move_time = travel_time(trip, movement_speed = movement_speed)
    visit_time = sum(poi.visit_time for poi in trip)
    return move_time + visit_time
end

"""
Calculates the cost to visit all POIs in a trip (in eurodollars)

parameters:
    trip::Daytrip:
        The daytrip
"""
cost(trip::Daytrip)::Float64 = sum(Array{Float64,1}([poi.cost for poi in trip]))

function optimize_trip(trip::Daytrip)
     # w[i,j] = distance from POI i to POI j
    w = [ distance(x,y) for x=trip, y=trip]
    n = length(trip)
    
    TSP = Model(GLPK.Optimizer)
    @variable(TSP, x[1:n,1:n], Bin)
    # minimize weght of all included edges
    @objective(TSP, Min, dot(w, x))
    # no self-loops should not be part of te solution
    @constraint(TSP, [i = 1:n], x[i,i] == 0)
    # Arcs with zero cost are not a part of the path as they do no exist
    @constraint(TSP, [i = 1:n, j = 1:n; w[i,j] == 0], x[i,j] == 0)
    # Every city has one incoming edge
    @constraint(TSP, [i = 1:n], sum(x[i,:]) == 1)
    # Every city has one outgoin edge
    @constraint(TSP, [j = 1:n], sum(x[:,j]) == 1)
    # Only a single tour should cover all cities
    @variable(TSP, u[2:n], Int)
    @constraint(TSP, [i = 2:n, j = 2:n; i != j], (u[i] - u[j] + n*x[i,j]) <= n-1)
    @constraint(TSP, [i = 2:n], 1 <= u[i] <= n-1);
    
    set_time_limit_sec(TSP, 1)
    
    try
        optimize!(TSP)
    catch e
        @warn "TSP solver failed"
        return trip
    end
    if termination_status(TSP) != JuMP.MathOptInterface.OPTIMAL
        return trip
    end
    
    solution = JuMP.value.(x)
    idx = 1
    new_trip = Daytrip()
    for _ in 1:n
        push!(new_trip, trip[idx])
        idx = argmax(solution[idx,:])
    end
    return new_trip
end

# implements Clarke-Wright savings heuristic for TSP problems for finding the optimal order for s
# https://www.cs.ubc.ca/~hutter/previous-earg/EmpAlgReadingGroup/TSP-JohMcg97.pdf
function optimize_trip_cw(s)
    if length(s) == 1
        return s
    end
    D = [distance(s[i],s[j]) for i in 1:length(s), j in 1:length(s)]
    loops = [[j] for j in 2:length(s)]
    while length(loops) > 1
        savings = [D[1,i[end]] + D[1,j[1]] - D[i[end],j[1]] for i in loops, j in loops]
        best = argmax(savings - (maximum(savings) * I))
        idx1, idx2 = best.I
        select = trues(length(loops))
        select[[idx1,idx2]] .= false
        loops = vcat(loops[select], [vcat(loops[idx1], loops[idx2])])
    end
    return s[vcat(1, loops[1])]
end


################################################################################
# DAYTRIP DESIGN AS AN MDP
################################################################################

@enum DaytripEditType ADD REMOVE NOOP

# ADD adds POI at index in MDP's POI list to the trip
# REMOVE removes the POI at index in daytrip
struct DaytripEdit
    edit_type::DaytripEditType
    index::Int32
end

function string(e::DaytripEdit)
    if e.edit_type == NOOP
        return "noop"
    elseif e.edit_type == REMOVE
        return "del " * string(e.index)
    elseif e.edit_type == ADD
        return "add " * string(e.index)
    else
        @warn "encountered unknown type in string conversion"
        return ""
    end
end

abstract type DaytripScheduleMDP <: MDP{Daytrip, DaytripEdit} end

POMDPs.discount(::DaytripScheduleMDP) = 1.0
POMDPs.initialstate(m::DaytripScheduleMDP) = Deterministic([m.home])
POMDPs.isterminal(::DaytripScheduleMDP, ::Daytrip) = false

function POMDPs.actions(m::DaytripScheduleMDP, s::Daytrip)
    ret = Array{DaytripEdit,1}()
    
    # NOOP action
    push!(ret, DaytripEdit(NOOP::DaytripEditType, 0))
    
    # REMOVE actions
    for i in 1:length(s)
        if s[i].category == HOME::POICategory
            continue
        end
        push!(ret, DaytripEdit(REMOVE::DaytripEditType, i))
    end
    
    # don't add ADD actions if the daytrip is full
    if duration(s) > MAX_DAY_LENGTH
        return ret
    end
    
    # ADD actions
    for i in 1:length(m.POIs)
        if m.POIs[i] in s
            continue
        end
        push!(ret, DaytripEdit(ADD::DaytripEditType, i))
    end
    
    return ret
end

POMDPs.reward(m::DaytripScheduleMDP, s::Daytrip, a::DaytripEdit, sp::Daytrip) = POMDPs.discount(m) * utility(m, sp) - utility(m, s)

function POMDPs.actionindex(m::DaytripScheduleMDP, a::DaytripEdit)::Int64
    if a.edit_type == DaytripDesignAssistance.NOOP
        return 1
    elseif a.edit_type == DaytripDesignAssistance.ADD
        return 1 + a.index
    else
        return 1 + a.index + length(m.POIs)
    end
end

"""
    Returns a multiple-hot encoded representation of state s suitable for ML.
"""
function featurize(m::DaytripScheduleMDP, s::Daytrip)::Array{Float32,1}
    sf = zeros(Float32, length(m.POIs))
    for (i,p) in enumerate(m.POIs)
        if p in s
            sf[i] = 1.0f0
        end
    end
    return sf
end

POMDPs.stateindex(m::DaytripScheduleMDP, s::Daytrip)::Int64 = sum(Int64(v == 1)*2^(i-1) for (i,v) in enumerate(featurize(m, s))) + 1


################################################################################
# USER MODELS
################################################################################

# This is a human conception of the daytrip design task. It differs from the real task in that the TSP problem that is part of trip planning is solved heuristically.
mutable struct HumanDaytripScheduleMDP <: DaytripScheduleMDP
    # state space definition
    POIs::Array{POI,1}
    home::POI
    
    # utility definition
    category_preferences::Array{Float64,1}
    travel_dislike::Float64
    cost_pref_mean::Float64
    cost_pref_std::Float64
    cost_weight::Float64
      
    function HumanDaytripScheduleMDP(
        POIs::Array{POI,1},
        home::POI,
        category_preferences::Array{Float64,1},
        travel_dislike::Float64,
        cost_pref_mean::Float64,
        cost_pref_std::Float64,
        cost_weight::Float64)

        @assert length(category_preferences) == (length(instances(POICategory))-1) "Incorrect number of category preferences (you should not specify preference for the HOME category!)."
        @assert all(category_preferences .<= 1.0) && all(category_preferences .>= -1.0) "Preferences for categories must be in [-1.0, 1.0]!"
        @assert (travel_dislike <= 1.0) && (travel_dislike >= 0.0) "Travel dislike must be in [0.0, 1.0]!"
        @assert (cost_pref_mean >= 0.0) "Cost preference mean parameter must be positive"
        @assert (cost_pref_std > 0.0) "Cost preference std parameter must be strictly positive"
        @assert (cost_weight <= 1.0) && (cost_weight >= 0.0) "Cost weight must be in [0.0, 1.0]!"
        new(POIs, home, vcat(category_preferences, 0.0), travel_dislike, cost_pref_mean, cost_pref_std, cost_weight)
    end
end

"""
Calculate the value of a trip (the utility of a design)

params:
    m::HumanDaytripScheduleMDP
        The human model for which we want  to evaluate the utility
    trip::Daytrip:
        The daytrip
"""
function utility(m::HumanDaytripScheduleMDP, trip::Daytrip)
    cost_factor = 1 - cdf(truncated(Normal(m.cost_pref_mean, m.cost_pref_std), 0.0, Inf), cost(trip))
    
    # calculate average fun per minute
    fun_factor = 0.0
    for poi in trip
        fun_factor += poi.visit_time * (m.category_preferences[Int(poi.category)] / 2 + 0.5)
    end
    fun_factor += travel_time(trip) * (0.5 - m.travel_dislike / 2)
    unspent_time = MAX_DAY_LENGTH - duration(trip)
    fun_factor += unspent_time * 0.5
    fun_factor /= MAX_DAY_LENGTH
    
    return (1-m.cost_weight) * fun_factor + m.cost_weight * cost_factor
end

# Humans can't solve the TSP at every planning step so here it is solved (iteratively) using a visual heuristic. If a POI is removed the order remains unchanged. If a POI is added it is inserted its location is chosen using the maximum angle heuristic.
function POMDPs.gen(m::HumanDaytripScheduleMDP, s::Daytrip, a::DaytripEdit, rng::AbstractRNG = Random.GLOBAL_RNG)
    sp = copy(s)
    if a.edit_type == NOOP::DaytripEditType
        # noop does nothing
    elseif a.edit_type == REMOVE::DaytripEditType
        deleteat!(sp, a.index)
    else
        if length(sp) == 1
            push!(sp, m.POIs[a.index])
        else
            # maximum angle heuristic
            loc = argmax([included_angle(sp[i], m.POIs[a.index], sp[i%length(sp)+1]) for i in 1:length(sp)])
            insert!(sp, loc+1, m.POIs[a.index])
        end
    end
    
    return (sp=sp, r=POMDPs.reward(m, s, a, sp), info=missing)
end

"""
    Returns the relevant paramters for a HumanDaytripScheduleMDP in format suitable for ML.

    NOTE: this implements some normalization which may not be suitable for your prior.
"""
featurize(mdp::HumanDaytripScheduleMDP)::Array{Float32,1} = convert(Array{Float32,1}, vcat(mdp.category_preferences, mdp.travel_dislike, mdp.cost_pref_std / 15.0, mdp.cost_pref_mean / 100.0, mdp.cost_weight))


"""
    Abstract type for user models based on an MCTS value estimator. Instances must have fields:
        world_model::HumanDaytripScheduleMDP
        value_estimator::MCTSPlanner
"""
abstract type MCTSUserModel end


#
# Limited Consideration User model
#
struct LCUserModel <: MCTSUserModel
    world_model::HumanDaytripScheduleMDP
    value_estimator::MCTSPlanner
    N_consider::Int64
    
    # trust in AI. User model's valuation v of AI's recommendation is increased by AI_trust * abs(v). Positive values encode trust, negative values distrust.
    AI_trust::Float64
    
    LCUserModel(m::HumanDaytripScheduleMDP, 
                planning_depth, 
                N_consider;
                AI_trust::Float64 = 0.0,
                MCTS_iterations = 500,
                MCTS_exploration_constant = 0.2) = new(m, 
                                                       MCTSPlanner(MCTSSolver(n_iterations = MCTS_iterations, 
                                                                              depth = planning_depth,
                                                                              exploration_constant = MCTS_exploration_constant,
                                                                              estimate_value = 0.0,
                                                                              reuse_tree = false), # reusing the tree gives weird results
                                                                   m),
                                                       N_consider, AI_trust)
end

function Base.show(io::IO, um::LCUserModel)
    println(io, "  HOME  : (", um.world_model.home.coord_x, ", ", um.world_model.home.coord_y, ")")
    println(io, "  #POIs : ", length(um.world_model.POIs))
    println(io, "-------------- PLANNING ------------")
    println(io, "  depth        : ", um.value_estimator.solver.depth)
    println(io, "  n_iterations : ", um.value_estimator.solver.n_iterations)
    println(io, "  exploration  : ", um.value_estimator.solver.exploration_constant)
    println(io, "  N_consider   : ", um.N_consider)
    println(io, "  AI_trust     : ", um.AI_trust)
    println(io, "-------------- UTILITY --------------")
    println(io, "  category_preferences : ", um.world_model.category_preferences)
    println(io, "  travel_dislike       : ", string(um.world_model.travel_dislike)[1:6])
    println(io, "  cost_pref_mean       : ", string(um.world_model.cost_pref_mean)[1:6])
    println(io, "  cost_pref_std        : ", string(um.world_model.cost_pref_std)[1:6])
    println(io, "  cost_weight          : ", string(um.world_model.cost_weight)[1:6])
end

"""
    Simulate the user model on design s with recommendation a_recommended. Set a_recommended = missing if you there is no recommendation
"""
function act(um::LCUserModel, s::Daytrip, a_recommended::Union{DaytripEdit,Missing} = missing; verbose = false)
    MCTS.plan!(um.value_estimator, s)
    a = DaytripEdit(NOOP::DaytripEditType, 0)
    best_q = reward(um.world_model, s, a, s)
    
    selected_idxs = shuffle(collect(1:length(actions(um.world_model, s))))[1:min(um.N_consider, length(actions(um.world_model, s)))]
    i = 1
    for san in children(get_state_node(um.value_estimator.tree, s))
        if i in selected_idxs
            i += 1
        else
            i += 1
            continue
        end
        if MCTS.q(san) >= best_q
            best_q = MCTS.q(san)
            a = MCTS.action(san)
        end
    end
    
    # follow recommended action if it is better
    if !ismissing(a_recommended)
        va = POMDPs.value(um.value_estimator, s, a_recommended)
        if verbose
            println("value of AI's choice (inflated): ", va, " (", va + abs(va) * um.AI_trust, "), human's choice: ", best_q)
        end
        if va + abs(va) * um.AI_trust >= best_q
            return a_recommended
        end
    end
    return a
end

function likelihood(um::LCUserModel, s::Daytrip, a::DaytripEdit, a_recommended::DaytripEdit)
    MCTS.plan!(um.value_estimator, s)
    
    q_values = Array{Float64,1}()
    noop_idx = 0
    a_recommended_idx = 0
    a_idx = 0
    for san in children(get_state_node(um.value_estimator.tree, s))
        a_p = MCTS.action(san)
        push!(q_values, MCTS.q(san))
        if a_p.edit_type == NOOP::DaytripEditType
            noop_idx = length(q_values)
        end
        if a_p == a_recommended
            a_recommended_idx = length(q_values)
        end
        if a_p == a
            a_idx = length(q_values)
        end
    end
    @assert noop_idx != 0
    @assert a_recommended_idx != 0
    @assert a_idx != 0
    
    p_selected = um.N_consider / length(q_values)
    probs = (1-p_selected) .^ (invperm(sortperm(q_values, rev=true)) .- 1) .* p_selected
    probs /= sum(probs)
    
    
    # all mass that is worse than no-op goes to no-op
    to_noop = q_values .< q_values[noop_idx]
    probs[noop_idx] += sum(probs[to_noop])
    probs[to_noop] .= 0.0
    
    # all probability mass for choices which resulted in less than the AI's recommendation go to the recommendation
    to_ai = q_values .<= q_values[a_recommended_idx] + um.AI_trust * abs(q_values[a_recommended_idx])
    s = sum(probs[to_ai])
    probs[to_ai] .= 0.0
    probs[a_recommended_idx] = s
    
    return probs[a_idx]
end

"""
Returns the policy of a user model as two lists: a list of the actions and a list of the probabilities of those actions
"""
function get_policy(um::LCUserModel, s::Daytrip, a_recommended::Union{DaytripEdit,Missing})
    MCTS.plan!(um.value_estimator, s)
    
    q_values = Array{Float64,1}()
    noop_idx = 0
    a_recommended_idx = 0
    a_idx = 0
    for san in children(get_state_node(um.value_estimator.tree, s))
        a_p = MCTS.action(san)
        push!(q_values, MCTS.q(san))
        if a_p.edit_type == NOOP::DaytripEditType
            noop_idx = length(q_values)
        end
        if !ismissing(a_recommended) && (a_p == a_recommended)
            a_recommended_idx = length(q_values)
        end
    end
    @assert noop_idx != 0
    @assert ismissing(a_recommended) || (a_recommended_idx != 0)
       
    p_selected = um.N_consider / length(q_values)
    probs = (1-p_selected) .^ (invperm(sortperm(q_values, rev=true)) .- 1) .* p_selected
    probs /= sum(probs)

    # all mass that is worse than no-op goes to no-op
    to_noop = q_values .< q_values[noop_idx]
    probs[noop_idx] += sum(probs[to_noop])
    probs[to_noop] .= 0.0
    
    if !ismissing(a_recommended)
        # all probability mass for choices which resulted in less than the AI's recommendation go to the recommendation
        to_ai = q_values .<= q_values[a_recommended_idx] + um.AI_trust * abs(q_values[a_recommended_idx])
        mass = sum(probs[to_ai])
        probs[to_ai] .= 0.0
        probs[a_recommended_idx] = mass
    end
    
    return actions(um.world_model, s), probs
end

"""
    Returns the relevant paramters for a LCUserModel in format suitable for ML.
"""
featurize(um::LCUserModel)::Array{Float32,1} = convert(Array{Float32,1}, vcat(um.value_estimator.solver.depth, um.N_consider, um.AI_trust, featurize(um.world_model)))


#
# Boltzmann Rational User model
#
struct BoltzmannUserModel <: MCTSUserModel
    world_model::HumanDaytripScheduleMDP
    value_estimator::MCTSPlanner
    multi_choice_optimality::Float64
    comparison_optimality::Float64
    
    function BoltzmannUserModel(m::HumanDaytripScheduleMDP,
                                planning_depth::Int64,
                                multi_choice_optimality::Float64,
                                comparison_optimality::Float64;
                                MCTS_iterations::Int64 = 500,
                                MCTS_exploration_constant = 0.2)
        @assert multi_choice_optimality >= 0.0
        @assert comparison_optimality >= 0.0
        return new(m,
                   MCTSPlanner(MCTSSolver(n_iterations = MCTS_iterations, 
                                          depth = planning_depth,
                                          exploration_constant = MCTS_exploration_constant,
                                          estimate_value = 0.0,
                                          reuse_tree = false), # reusing the tree gives weird results
                               m),
                   multi_choice_optimality,
                   comparison_optimality)
    end
end

function Base.show(io::IO, um::BoltzmannUserModel)
    println(io, "  HOME  : (", um.world_model.home.coord_x, ", ", um.world_model.home.coord_y, ")")
    println(io, "  #POIs : ", length(um.world_model.POIs))
    println(io, "-------------- PLANNING ------------")
    println(io, "  depth                   : ", um.value_estimator.solver.depth)
    println(io, "  n_iterations            : ", um.value_estimator.solver.n_iterations)
    println(io, "  exploration             : ", um.value_estimator.solver.exploration_constant)
    println(io, "  multi_choice_optimality : ", string(um.multi_choice_optimality)[1:6])
    println(io, "  comparison_optimality   : ", string(um.comparison_optimality)[1:6])
    println(io, "-------------- UTILITY --------------")
    println(io, "  category_preferences : ", um.world_model.category_preferences)
    println(io, "  travel_dislike       : ", string(um.world_model.travel_dislike)[1:6])
    println(io, "  cost_pref_mean       : ", string(um.world_model.cost_pref_mean)[1:6])
    println(io, "  cost_pref_std        : ", string(um.world_model.cost_pref_std)[1:6])
    println(io, "  cost_weight          : ", string(um.world_model.cost_weight)[1:6])
end

"""
Returns the policy of a user model as two lists: a list of the actions and a list of the probabilities of those actions
"""
function get_policy(um::BoltzmannUserModel, s::Daytrip, a_recommended::Union{DaytripEdit,Missing})
    MCTS.plan!(um.value_estimator, s)

    q_values = Array{Float64,1}()
    s_actions = Array{DaytripEdit,1}()
    noop_idx = 0
    a_recommended_idx = 0
    for san in children(get_state_node(um.value_estimator.tree, s))
        a_p = MCTS.action(san)
        push!(q_values, MCTS.q(san))
        push!(s_actions, a_p)
        
        if a_p.edit_type == NOOP::DaytripEditType
            noop_idx = length(q_values)
        end
        if !ismissing(a_recommended) && (a_p == a_recommended)
            a_recommended_idx = length(q_values)
        end
    end
    @assert noop_idx != 0
    @assert ismissing(a_recommended) || (a_recommended_idx != 0)
    
    # normalize Q values, otherwise the softmaxs's parameters have to be scaled constantly
    q_values .+= mean(q_values)
    q_values ./= std(q_values)

    # prior selection probabilities
    probs = exp.(um.multi_choice_optimality .* q_values)
    probs ./= sum(probs)

    # Boltzmann-rational switch to recommended action
    # utility of switching is increase in Q-value from switching, utility of not switching is 0.
    if !ismissing(a_recommended)
        switch_prob = exp.(um.comparison_optimality .* (q_values[a_recommended_idx] .- q_values)) ./ (exp.(um.comparison_optimality .* (q_values[a_recommended_idx] .- q_values)) .+ 1.0)
        switched_probs = zeros(length(q_values))
        switched_probs[a_recommended_idx] = sum(probs .* switch_prob)
        probs = (1.0 .- switch_prob) .* probs + switched_probs
    end

    # Boltzmann-rational switch to NOOP
    switch_prob = exp.(um.comparison_optimality .* (q_values[noop_idx] .- q_values)) ./ (exp.(um.comparison_optimality .* (q_values[noop_idx] .- q_values)) .+ 1.0)
    switched_probs = zeros(length(q_values))
    switched_probs[noop_idx] = sum(probs .* switch_prob)
    probs = (1.0 .- switch_prob) .* probs + switched_probs

    return s_actions, probs
end

"""
    Simulate the user model on design s with recommendation a_recommended. Set a_recommended = missing if you there is no recommendation
"""
function act(um::BoltzmannUserModel, s::Daytrip, a_recommended::Union{DaytripEdit,Missing} = missing; verbose = false)
    s_actions, probs = get_policy(um, s, a_recommended)
    return s_actions[rand(Categorical(probs))]
end

function likelihood(um::BoltzmannUserModel, s::Daytrip, a::DaytripEdit, a_recommended::DaytripEdit)
    s_actions, probs = get_policy(um, s, a_recommended)
    for (i,a_p) in enumerate(s_actions)
        if a_p == a
            return probs[i]
        end
    end
    return 0.0
end

"""
    Returns the relevant paramters for a BoltzmannUserModel in format suitable for ML.
"""
featurize(um::BoltzmannUserModel)::Array{Float32,1} = convert(Array{Float32,1}, vcat(um.value_estimator.solver.depth, um.multi_choice_optimality, um.comparison_optimality, featurize(um.world_model)))


################################################################################
# USER MODEL SPECIFICATIONS
################################################################################

"""
    Abstract type for user models specifications. These capture the parameters of the user model but not the data (POIs mainly). They are used to reduce memory usage.
"""
abstract type MCTSUserModelSpecification end


struct LCUserModelSpecification <: MCTSUserModelSpecification
    # utility:
    category_preferences::Array{Float64,1}
    travel_dislike::Float64
    cost_pref_mean::Float64
    cost_pref_std::Float64
    cost_weight::Float64

    # user model:
    planning_depth::Int64
    n_iterations::Int64
    MCTS_exploration_constant::Float64
    N_consider::Int64
    AI_trust::Float64
end

"""
    Instantiate a user model based on the given spec.
"""
function instantiate(um_spec::LCUserModelSpecification, home::POI, POIs::Array{POI, 1})
    return LCUserModel(HumanDaytripScheduleMDP(POIs,
                                               home,
                                               um_spec.category_preferences,
                                               um_spec.travel_dislike,
                                               um_spec.cost_pref_mean,
                                               um_spec.cost_pref_std,
                                               um_spec.cost_weight),
    um_spec.planning_depth, um_spec.N_consider, MCTS_iterations = um_spec.n_iterations, MCTS_exploration_constant = um_spec.MCTS_exploration_constant, AI_trust = um_spec.AI_trust)
end

function Base.show(io::IO, um_spec::LCUserModelSpecification)
    println(io, "     LCUserModel Specification")
    println(io, "-------------- PLANNING ------------")
    println(io, "  depth                : ", um_spec.planning_depth)
    println(io, "  n_iterations         : ", um_spec.n_iterations)
    println(io, "  exploration constant : ", um_spec.MCTS_exploration_constant)
    println(io, "  N_consider           : ", um_spec.N_consider)
    println(io, "  AI_trust             : ", um_spec.AI_trust)
    println(io, "-------------- UTILITY --------------")
    println(io, "  category_preferences : ", um_spec.category_preferences)
    println(io, "  travel_dislike       : ", string(um_spec.travel_dislike)[1:6])
    println(io, "  cost_pref_mean       : ", string(um_spec.cost_pref_mean)[1:6])
    println(io, "  cost_pref_std        : ", string(um_spec.cost_pref_std)[1:6])
    println(io, "  cost_weight          : ", string(um_spec.cost_weight)[1:6])
end


struct BoltzmannUserModelSpecification <: MCTSUserModelSpecification
    # utility:
    category_preferences::Array{Float64,1}
    travel_dislike::Float64
    cost_pref_mean::Float64
    cost_pref_std::Float64
    cost_weight::Float64

    # user model:
    planning_depth::Int64
    n_iterations::Int64
    MCTS_exploration_constant::Float64
    multi_choice_optimality::Float64
    comparison_optimality::Float64
end

"""
    Instantiate a user model based on the given spec.
"""
function instantiate(um_spec::BoltzmannUserModelSpecification, home::POI, POIs::Array{POI, 1})
    return BoltzmannUserModel(HumanDaytripScheduleMDP(POIs,
                                               home,
                                               um_spec.category_preferences,
                                               um_spec.travel_dislike,
                                               um_spec.cost_pref_mean,
                                               um_spec.cost_pref_std,
                                               um_spec.cost_weight),
    um_spec.planning_depth, um_spec.multi_choice_optimality, um_spec.comparison_optimality, MCTS_iterations = um_spec.n_iterations, MCTS_exploration_constant = um_spec.MCTS_exploration_constant)
end

function Base.show(io::IO, um_spec::BoltzmannUserModelSpecification)
    println(io, "     LCUserModel Specification")
    println(io, "-------------- PLANNING ------------")
    println(io, "  depth                   : ", um_spec.planning_depth)
    println(io, "  n_iterations            : ", um_spec.n_iterations)
    println(io, "  exploration constant    : ", um_spec.MCTS_exploration_constant)
    println(io, "  multi_choice_optimality : ", string(um_spec.multi_choice_optimality)[1:6])
    println(io, "  comparison_optimality   : ", string(um_spec.comparison_optimality)[1:6])
    println(io, "-------------- UTILITY --------------")
    println(io, "  category_preferences : ", um_spec.category_preferences)
    println(io, "  travel_dislike       : ", string(um_spec.travel_dislike)[1:6])
    println(io, "  cost_pref_mean       : ", string(um_spec.cost_pref_mean)[1:6])
    println(io, "  cost_pref_std        : ", string(um_spec.cost_pref_std)[1:6])
    println(io, "  cost_weight          : ", string(um_spec.cost_weight)[1:6])
end


################################################################################
# ASSISTANCE MDP
################################################################################

mutable struct DaytripScheduleAssistanceMDP <: DaytripScheduleMDP
    user_model::MCTSUserModel
        
    # these are technically already present in user_model but they're stored here too
    POIs::Array{POI,1}
    home::POI
    
    discounting::Float64
    
    optimizer::Symbol
    
    DaytripScheduleAssistanceMDP(m::MCTSUserModel, discounting::Float64) = new(m, m.world_model.POIs, m.world_model.home, discounting, :CW)
end

POMDPs.discount(m::DaytripScheduleAssistanceMDP) = m.discounting

utility(m::DaytripScheduleAssistanceMDP, s::Daytrip) = utility(m.user_model.world_model, s)

function POMDPs.gen(m::DaytripScheduleAssistanceMDP, s::Daytrip, a::DaytripEdit, rng::AbstractRNG = Random.GLOBAL_RNG; verbose = false, optimizer = :default)
    if optimizer == :default
        optimizer = m.optimizer
    end
    
    sp = copy(s)
    
    a_sup = act(m.user_model, s, a; verbose = verbose)
    
    if a_sup.edit_type == NOOP::DaytripEditType
        # noop does nothing
    elseif a_sup.edit_type == REMOVE::DaytripEditType
        deleteat!(sp, a_sup.index)
        if optimizer == :LP
            sp = optimize_trip(sp)
        elseif optimizer == :CW
            sp = optimize_trip_cw(sp)
        else
            @error "specified TSP solver for planning does not exist"
        end
    else
        push!(sp, m.POIs[a_sup.index])
        if optimizer == :LP
            sp = optimize_trip(sp)
        elseif optimizer == :CW
            sp = optimize_trip_cw(sp)
        else
            @error "specified TSP solver for planning does not exist"
        end
    end
    
    return (sp=sp, r=POMDPs.reward(m, s, a_sup, sp), info=Dict("user action" => a_sup))
end

function POMDPs.transition(m::DaytripScheduleAssistanceMDP, s::Daytrip, a::Union{DaytripEdit,Missing}; optimizer = :default)
    if optimizer == :default
        optimizer = m.optimizer
    end
    
    H_actions, probs = get_policy(m.user_model, s, a)

    next_states = Array{Daytrip,1}()
    sizehint!(next_states, length(probs))

    for a_H in H_actions
        sp = copy(s)

        if a_H.edit_type == NOOP::DaytripEditType
            # noop does nothing
        elseif a_H.edit_type == REMOVE::DaytripEditType
            deleteat!(sp, a_H.index)
            if optimizer == :LP
                sp = optimize_trip(sp)
            elseif optimizer == :CW
                sp = optimize_trip_cw(sp)
            else
                @error "specified TSP solver for planning does not exist"
            end
        else
            push!(sp, m.POIs[a_H.index])
            if optimizer == :LP
                sp = optimize_trip(sp)
            elseif optimizer == :CW
                sp = optimize_trip_cw(sp)
            else
                @error "specified TSP solver for planning does not exist"
            end
        end
        push!(next_states, sp)
    end
    return SparseCat(next_states, probs)
end


#
# Rollout Policies
#

"""
Under this rollout policy the AI will simply recommend the action with highest Q-value under the user's world model.
"""
mutable struct RecommendOptimal <: Policy
    user_model::MCTSUserModel
end

function POMDPs.action(p::RecommendOptimal, s::Daytrip)
    return action(p.user_model.value_estimator, s)
end

end # DaytripDesignAssistance
