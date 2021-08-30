#  Copyright 2017-21, Oscar Dowson.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

using SDDP
using Gurobi
using BiObjectiveSDDP
using Random

BiObjectiveSDDP.include_gurobi_specific_functions()

const GUROBI_ENV = Gurobi.Env()

function create_model()
    model = SDDP.LinearPolicyGraph(
        stages = 2,
        lower_bound = 0.0,
        optimizer = with_optimizer(
            Gurobi.Optimizer,
            GUROBI_ENV,
            OutputFlag = 0,
        ),
    ) do sp, t
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
        SDDP.add_objective_state(
            sp,
            initial_value = 1.0,
            lower_bound = 0.0,
            upper_bound = 1.0,
            lipschitz = 10.0,
        ) do y, ω
            return y
        end
        SDDP.parameterize(sp, [nothing]) do ω
            λ = SDDP.objective_state(sp)
            @stageobjective(sp, λ * objective_1 + (1 - λ) * objective_2)
        end
    end
    return model
end

import Logging
Logging.global_logger(Logging.ConsoleLogger(stdout, Logging.Debug))

Random.seed!(1)

bounds_for_reporting = Tuple{Float64,Float64,Float64}[]

model = create_model()
lower_bound, weights, bounds = BiObjectiveSDDP.bi_objective_sddp(
    model,
    with_optimizer(Gurobi.Optimizer, GUROBI_ENV, OutputFlag = 0);
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
                with_optimizer(Gurobi.Optimizer, GUROBI_ENV, OutputFlag = 0);
                global_lower_bound = 0.0,
                lambda_minimum_step = 1e-4,
                lambda_atol = 1e-4,
            )
            lower_bound, _, _ = BiObjectiveSDDP.surrogate_lower_bound(
                model,
                with_optimizer(Gurobi.Optimizer, GUROBI_ENV, OutputFlag = 0);
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
