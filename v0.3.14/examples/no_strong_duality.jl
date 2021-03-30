#  Copyright 2017-21, Oscar Dowson.                                     #src
#  This Source Code Form is subject to the terms of the Mozilla Public  #src
#  License, v. 2.0. If a copy of the MPL was not distributed with this  #src
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.             #src

# # No strong duality

# This example is interesting, because strong duality doesn't hold for the
# extensive form (see if you can show why!), but we still converge.

using SDDP, GLPK, Test

function no_strong_duality()
    model = SDDP.PolicyGraph(
        SDDP.Graph(:root, [:node], [(:root => :node, 1.0), (:node => :node, 0.5)]),
        optimizer = GLPK.Optimizer,
        lower_bound = 0.0,
    ) do sp, t
        @variable(sp, x, SDDP.State, initial_value = 1.0)
        @stageobjective(sp, x.out)
        @constraint(sp, x.in == x.out)
    end
    SDDP.train(model, iteration_limit = 20, log_frequency = 10)
    @test SDDP.calculate_bound(model) ≈ 2.0 atol = 1e-8
    return
end

no_strong_duality()
