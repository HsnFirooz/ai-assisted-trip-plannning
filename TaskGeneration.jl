using Distributions
using Random

function sample_spec_from_prior(::Type{LCUserModel}; n_iterations::Int64 = 500, exploration_constant::Float64 = 0.2)
    pcat_weights = [0.18, 0.20, 0.14, 0.16, 0.32]
    
    category_preferences = rand(truncated(Normal(0.5, 1.0), -1.0, 1.0), length(instances(POICategory))-1)
    category_preferences[rand(Categorical(pcat_weights ./ sum(pcat_weights)))] = 1.0
    travel_dislike = rand(Beta(1, 4))
    cost_pref_std = rand(Uniform(5.0, 15.0))
    cost_pref_mean = rand(Normal(100.0, 25.0)) + 4 * cost_pref_std
    cost_weight = rand(Uniform(0.0, 0.4))
    planning_depth = rand(Categorical([0.25, 0.35, 0.25, 0.15]))
    N_consider = rand(Categorical([0.0, 0.04, 0.10, 0.20, 0.20, 0.15, 0.10, 0.08, 0.07, 0.06]))
    AI_trust = 0.0
    
    return LCUserModelSpecification(category_preferences, travel_dislike, cost_pref_mean, cost_pref_std, cost_weight, planning_depth, n_iterations, exploration_constant, N_consider, AI_trust)
end

function sample_spec_from_prior(::Type{BoltzmannUserModel}; n_iterations::Int64 = 500, exploration_constant::Float64 = 0.2)
    pcat_weights = [0.18, 0.20, 0.14, 0.16, 0.32]
    
    category_preferences = rand(truncated(Normal(0.5, 1.0), -1.0, 1.0), length(instances(POICategory))-1)
    category_preferences[rand(Categorical(pcat_weights ./ sum(pcat_weights)))] = 1.0
    travel_dislike = rand(Beta(1, 4))
    cost_pref_std = rand(Uniform(5.0, 15.0))
    cost_pref_mean = rand(Normal(100.0, 25.0)) + 4 * cost_pref_std
    cost_weight = rand(Uniform(0.0, 0.4))
    planning_depth = rand(Categorical([0.11, 0.19, 0.33, 0.24, 0.13]))
    multi_choice_optimality = rand(Frechet(3.0, 1.25))
    comparison_optimality = 1.5 * multi_choice_optimality
    
    return BoltzmannUserModelSpecification(category_preferences, travel_dislike, cost_pref_mean, cost_pref_std, cost_weight, planning_depth, n_iterations, exploration_constant, multi_choice_optimality, comparison_optimality)
end


"""
Generates a city of points of interest.

paramters:
    N_POIs::Int64:
        number of points of interest to generate
    x_limits::Float64:
        limit from (0,0) on the size of the square city limits within POIs are placed. Measured in km.
        default: 5.0
    y_limits::Float64:
        limit from (0,0) on the size of the square city limits within POIs are placed. Measured in km.
        default: 5.0
"""
function generate_city(N_POIs::Int64 = 100; x_limits::Float64 = 5.0, y_limits::Float64 = 5.0)
    location_distribution = MvNormal(2, 1.15)
    category_distribution = Categorical([0.06, 0.14, 0.38, 0.10, 0.32])
    
    cityPOIs = Array{POI,1}()
    for _ in 1:N_POIs
        c_x = Inf
        c_y = Inf
        while (c_x > x_limits) || (c_x < -x_limits) || (c_y > y_limits) || (c_y < -y_limits)
            c_x, c_y = rand(location_distribution, 1)
        end
        
        c_category = POICategory(rand(category_distribution, 1)[1])
        c_cost = Inf
        c_visit_time = Inf
        if c_category == FINEARTS::POICategory
            c_cost = rand(truncated(Normal(15.0, 3.0), 0.0, Inf))
            c_visit_time = rand(truncated(Normal(30.0, 20.0), 0.0, 100.0))
        elseif c_category == POPCULTURE::POICategory
            c_cost = rand(truncated(Normal(20.0, 3.0), 0.0, Inf))
            c_visit_time = rand(truncated(Normal(30.0, 20.0), 0.0, 100.0))
        elseif c_category == SIGHTSEEING::POICategory
            c_cost = 0.0
            c_visit_time = rand(truncated(Normal(10.0, 5.0), 0.0, 20.0))
        elseif c_category == HISTORICHOUSE::POICategory
            c_cost = rand(truncated(Normal(15.0, 3.0), 0.0, Inf))
            c_visit_time = rand(truncated(Normal(30.0, 20.0), 0.0, 70.0))
        elseif c_category == SHOPPING::POICategory
            c_cost = 0.0
            c_visit_time = rand(truncated(Normal(30.0, 10.0), 10.0, 45.0))
        else
            @warn "unknown POICategory encountered in generate_city"
        end
        
        p = POI(c_x, c_y, c_category, c_visit_time, c_cost)
        push!(cityPOIs, p)
    end
    return cityPOIs
end