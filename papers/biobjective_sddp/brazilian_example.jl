#  Copyright 2017-21, Oscar Dowson.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

using SDDP
using Gurobi
using Plots
using BiObjectiveSDDP

BiObjectiveSDDP.include_gurobi_specific_functions()

const OBJ_1_SCALING = 0.01
const OBJ_2_SCALING = 0.1

function create_model()
    include("brazilian_data.jl")
    env = Gurobi.Env()
    model = SDDP.LinearPolicyGraph(
        stages = 12,
        lower_bound = 0.0,
        optimizer = () -> Gurobi.Optimizer(env),
    ) do sp, t
        set_silent(sp)
        month = t % 12 == 0 ? 12 : t % 12  # Year to month conversion.
        @variables(
            sp,
            begin
                # State variables.
                0 <= storedEnergy[i = 1:4] <= storedEnergy_ub[i],
                (SDDP.State, initial_value = storedEnergy_initial[i])
                # Control variables.
                0 <= spillEnergy[i = 1:4]
                0 <= hydroGeneration[i = 1:4] <= hydro_ub[i]
                thermal_lb[i][j] <=
                thermal[i = 1:4, j = 1:N_THERMAL[i]] <=
                thermal_ub[i][j]
                0 <= exchange[i = 1:5, j = 1:5] <= exchange_ub[i][j]
                0 <=
                deficit[i = 1:4, j = 1:4] <=
                demand[month][i] * deficit_ub[j]
                # Dummy variables for helpers.
                inflow[i = 1:4]
            end
        )
        @constraints(
            sp,
            begin
                # Model constraints.
                [i = 1:4],
                sum(deficit[i, :]) +
                hydroGeneration[i] +
                sum(thermal[i, j] for j in 1:N_THERMAL[i]) +
                sum(exchange[:, i]) - sum(exchange[i, :]) ==
                demand[month][i]
                [i = 1:4],
                storedEnergy[i].out + spillEnergy[i] + hydroGeneration[i] -
                storedEnergy[i].in == inflow[i]
                sum(exchange[:, 5]) == sum(exchange[5, :])
            end
        )
        Ω = if t == 1
            [inflow_initial]
        else
            r = (t - 1) % 12 == 0 ? 12 : (t - 1) % 12
            [
                [scenarios[i][r][ω] for i in 1:4] for
                ω in 1:length(scenarios[1][r])
            ]
        end
        @expressions(
            sp,
            begin
                objective_1,
                OBJ_1_SCALING *
                sum(deficit_obj[i] * sum(deficit[i, :]) for i in 1:4)
                objective_2,
                OBJ_2_SCALING * sum(
                    thermal_obj[i][j] * thermal[i, j] for i in 1:4 for
                    j in 1:N_THERMAL[i]
                )
            end
        )
        SDDP.initialize_biobjective_subproblem(sp)
        SDDP.parameterize(sp, Ω) do ω
            JuMP.fix.(inflow, ω)
            return SDDP.set_biobjective_functions(sp, objective_1, objective_2)
        end
    end
    return model
end

function simulate_policy(model, keys)
    simulations = Dict()
    for λ in keys
        BiObjectiveSDDP.set_scalarizing_weight(model, λ)
        simulations[λ] = SDDP.simulate(
            model,
            1000,
            [:storedEnergy, :objective_1, :objective_2],
        )
    end
    return simulations
end

function extract_objectives(simulation)
    obj_1 = [sum(s[:objective_1] for s in sim) for sim in simulation]
    obj_2 = [sum(s[:objective_2] for s in sim) for sim in simulation]
    return obj_1, obj_2
end

function plot_objective_space(simulations, simulation_weights)
    plot(title = "Objective Space")
    for λ in simulation_weights
        scatter!(
            extract_objectives(simulations[λ])...,
            label = "\\lambda = $(λ)",
            alpha = 0.4,
        )
    end
    p = plot!(xlabel = "Deficit cost", ylabel = "Thermal cost")
    savefig("objective_space.pdf")
    return p
end

function plot_weight_space(weights, bounds, simulations)
    plot(weights, bounds, label = "")
    for (λ, sim) in simulations
        obj_1, obj_2 = extract_objectives(simulations[λ])
        weighted_sum = λ .* obj_1 .+ (1 - λ) .* obj_2
        scatter!(
            fill(λ, length(weighted_sum)),
            weighted_sum,
            alpha = 0.4,
            label = "",
        )
    end
    p = plot!(xlabel = "Weight \\lambda", ylabel = "Weighted-sum")
    savefig("weight_space.pdf")
    return p
end

# function plot_publication(simulations, simulation_weights)
#     plts = [
#         SDDP.publication_plot(simulations[λ]; title = "λ = $(λ)") do data
#             return sum(data[:storedEnergy][i].out for i in 1:4)
#         end
#         for λ in simulation_weights
#     ]
#     plot(plts...,
#         xlabel = "Stage",
#         layout = (1, length(keys)),
#         margin = 5Plots.mm,
#         size = (1000, 300),
#         ylim = (0, 2.5e5),
#         ylabel = "Stored energy"
#     )
# end

import Logging
Logging.global_logger(Logging.ConsoleLogger(stdout, Logging.Debug))

model = create_model()
lower_bound, weights, bounds = BiObjectiveSDDP.bi_objective_sddp(
    model,
    with_optimizer(Gurobi.Optimizer, GUROBI_ENV, OutputFlag = 0);
    # BiObjectiveSDDP kwargs ...
    bi_objective_minor_iteration_limit = 60,
    bi_objective_lambda_atol = 1e-3,
    bi_objective_lower_bound = 0.0,
    bi_objective_major_iteration_burn_in = 20,
    # SDDP.jl kwargs ...
    print_level = 0,
    stopping_rules = [SDDP.BoundStalling(3, 1.0), SDDP.IterationLimit(50)],
)
simulation_weights = [0.1, 0.7, 0.9]
simulations = simulate_policy(model, simulation_weights);
plot_objective_space(simulations, simulation_weights)
plot_weight_space(weights, bounds, simulations)

function save_lower_bound_to_dat(weights, bounds)
    open("bounds.dat", "w") do io
        for (w, b) in zip(weights, bounds)
            println(io, "$(w)  $(b)")
        end
    end
end

function save_simulations_to_dat(simulations, simulation_weights)
    A = Matrix{Float64}(
        undef,
        length(simulations[first(simulation_weights)]),
        2 * length(simulation_weights),
    )
    for (i, weight) in enumerate(simulation_weights)
        obj_1, obj_2 = extract_objectives(simulations[weight])
        A[:, 2*i-1] .= obj_1
        A[:, 2*i] .= obj_2
    end
    open("simulations.dat", "w") do io
        for i in 1:size(A, 1)
            print(io, A[i, 1])
            for j in 2:size(A, 2)
                print(io, "  ", A[i, j])
            end
            println(io)
        end
    end
end
