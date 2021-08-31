#  Copyright 2017-21, Oscar Dowson.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

include(joinpath(@__DIR__, "BiObjectiveSDDP.jl"))
using .BiObjectiveSDDP

using SDDP
import Gurobi
import Random

const GUROBI_ENV = Gurobi.Env()

function create_model()
    model = SDDP.LinearPolicyGraph(
        stages = 2,
        lower_bound = 0.0,
        optimizer = () -> Gurobi.Optimizer(GUROBI_ENV),
    ) do sp, t
        set_silent(sp)
        @variable(sp, x >= 0, SDDP.State, initial_value = 0.0)
        if t == 1
            @expression(sp, objective_1, 2 * x.out)
            @expression(sp, objective_2, x.out)
        else
            @variable(sp, y >= 0)
            @constraints(sp, begin
                1.0 * x.in + y >= 1.00
                0.5 * x.in + y >= 0.75
                y >= 0.25
            end)
            @expression(sp, objective_1, y)
            @expression(sp, objective_2, 3 * y)
        end
        SDDP.initialize_biobjective_subproblem(sp)
        SDDP.parameterize(sp, [nothing]) do ω
            SDDP.set_biobjective_functions(sp, objective_1, objective_2)
        end
    end
    return model
end

Random.seed!(1)

bounds_for_reporting = Tuple{Float64,Float64,Float64}[]

model = create_model()
lower_bound, weights, bounds = BiObjectiveSDDP.bi_objective_sddp(
    model,
    () -> Gurobi.Optimizer(GUROBI_ENV);
    # BiObjectiveSDDP kwargs ...
    bi_objective_sddp_iteration_limit = 20,
    bi_objective_lower_bound = 0.0,
    bi_objective_lambda_update_method = BiObjectiveSDDP.MinimumUpdate(),
    bi_objective_lambda_atol = 1e-6,
    bi_objective_major_iteration_burn_in = 1,
    bi_objective_post_train_callback = (model::SDDP.PolicyGraph, λ) ->
        begin
            upper_bound = BiObjectiveSDDP.surrogate_upper_bound(
                model,
                () -> Gurobi.Optimizer(GUROBI_ENV);
                global_lower_bound = 0.0,
                lambda_minimum_step = 1e-4,
                lambda_atol = 1e-4,
            )
            lower_bound, _, _ = BiObjectiveSDDP.surrogate_lower_bound(
                model,
                () -> Gurobi.Optimizer(GUROBI_ENV);
                global_lower_bound = 0.0,
                lambda_minimum_step = 1e-4,
                lambda_atol = 1e-4,
            )
            push!(bounds_for_reporting, (λ, lower_bound, upper_bound))
        end,
    # SDDP.jl kwargs ...
    iteration_limit = 1,
    print_level = 0,
)

hcat(
    1:length(bounds_for_reporting),
    [b[1] for b in bounds_for_reporting],
    [b[2] for b in bounds_for_reporting],
    [b[3] for b in bounds_for_reporting],
)
