# module moduleODEnet_sp

using DifferentialEquations, DiffEqSensitivity, CSV, DataFrames, TensorBoardLogger, Flux, Suppressor, Random, StatsBase
using Logging: with_logger
import DiffEqFlux.multiple_shoot
import DiffEqFlux.group_ranges
include("ArgsODEnet.jl") # used by several modules
include("moduleODEnet.jl") 
import .moduleODEnet.gather_ODE_info
import .moduleODEnet.conSTRUCTor
import .moduleODEnet.reconstruct
import .moduleODEnet.subject
import .moduleODEnet.Data

# import .moduleODEnet.update_stressor_level!
"""
`update_stressor_level!()` is the affect function of this model. No condtion is needed since the `PresetTimeCallback` is used, see [here](https://diffeq.sciml.ai/latest/features/callback_functions/#PresetTimeCallback) for more info.
```
"""
function update_stressor_level!(integrator)
    if W[2,:][findfirst(arg -> arg == integrator.t, t)] == 1.0 
        integrator.u[2] = z_sl[findfirst(arg -> arg == integrator.t, t)]
    end
end


"""
split_observation_period_at_first_baseline() is a non-generic function that splits a the MARP visit number variable (mnpvisno) in before and after 2n baseline visit (indicated in the data by `mnpvisno == 1` for the second time).
A more generic way would be to split after every baseline (with `findall` and `while`), however, it is not sure we will get preprocessed data from the third baseline.
# Examples
```julia
julia> X1,X2,t, W, obs = d[findfirst(x -> x==3872, nonsparse_ids)][1] # 3872 has an interesting missing pattern
julia> individual_ranges = split_observation_period_at_first_baseline(obs,W)
2-element Vector{UnitRange{Int64}}:
 1:8
 8:14
```
"""
function split_observation_period_at_first_baseline(mnpvisno::AbstractArray, W::AbstractArray)

    if sum(x -> x == 1.0, mnpvisno[2:end]) >= 1.0 # more than one baseline observation after first one? sound silly but sometimes the first baseline obs is missing. This presereves some long TS
        
        # T_B2 = (findfirst(x -> x == 1, mnpvisno[2:end]) + 1)# +1 is necessary because you look only for mnpvisno[2:end], i.e., cutting one obs
        T_B2 = (findfirst(x -> x == 1, mnpvisno[2:end]))

        length_sp1 = 1:T_B2 # include B2 in sp1
        length_sp2 = T_B2:length(mnpvisno)
        
        if size(length_sp1,1) < 2 || size(length_sp2,1) < 2 || sum(W[1,length_sp1]) < 2 || sum(W[1,length_sp2]) < 2 || sum(W[2,length_sp1]) == 2 || sum(W[2,length_sp2]) == 2 # in case it sp1 or sp2 is only one obs long
            return [1:length(mnpvisno)] # full if sp2 is too sh
        else
            return [length_sp1, length_sp2] # split
        end
    else
         return [1:length(mnpvisno)] # full if no 2nd baseline
    end
end

"""
`calc_sp_solutions()` remakes the problem and calculates/returns the solutions for the supplied ranges of one respondent. It is used in training.
Using multiple dispatch, this can be also used for plotting when additionally providing mytsteps.
# Arguments
- `t::AbstractArray`: 1×T matrix of study time in continuous time.
- `cur_ranges::AbstractArray`: Current ranges of the respondent
- `odep::AbstractArray`
- `prob::ODEProblem`
- `mytsteps::Float64`: (optional) steps for saving the ODE solutions, e.g. 0.1 for plotting
- `solver::DiffEqBase.AbstractODEAlgorithm = Tsit5()`
- `kwargs`
"""
function calc_sp_solutions(t::AbstractArray, cur_ranges::AbstractArray, odep::AbstractArray, prob::ODEProblem; solver::DiffEqBase.AbstractODEAlgorithm = Tsit5(), kwargs...)
    sols = [
        solve(
            remake(prob;
                p = odep[findfirst(x->x==rg, cur_ranges)][1:4], 
                tspan=(t[first(rg)], t[last(rg)]),
                u0=odep[findfirst(x->x==rg, cur_ranges)][5:6],
            ),
            solver;
            saveat = t[rg],
            kwargs...
        ) for rg in cur_ranges
    ]
    return sols
end
function calc_sp_solutions(t::AbstractArray, cur_ranges::AbstractArray, odep::AbstractArray, prob::ODEProblem, mytsteps::Float64; solver::DiffEqBase.AbstractODEAlgorithm = Tsit5(), kwargs...)
    sols = [
        solve(
            remake(prob;
                p = odep[findfirst(x->x==rg, cur_ranges)][1:4], 
                tspan=(t[first(rg)], t[last(rg)]),
                u0=odep[findfirst(x->x==rg, cur_ranges)][5:6],
            ),
            solver;
            saveat = mytsteps,
            kwargs...
        ) for rg in cur_ranges
    ]
    return sols
end
"""
`calc_solution()` remakes the problem and calculates/returns the solutions for the supplied ranges of one respondent. It is used in data generation of the simulation.
# Arguments
- `t`: 1×T matrix of study time in continuous time.
- `ic`: 1x2 matrix of the initial conditions: 
- `odep`: ODE parameters
- `prob`: ODE problem
- `mysaveat`: expects vector or range to save solution at
- `solver:` default is `Tsit5()`
- `kwargs`: anything else you want to supply to the `solver()`
# Note
Is mysaveat and t really necessary?
"""
function calc_solution(t::AbstractArray, ic::AbstractArray, odep::AbstractArray, prob::ODEProblem, mysaveat::AbstractArray; solver::DiffEqBase.AbstractODEAlgorithm = Tsit5(), kwargs...) 
    solve(remake(prob;
                p = odep, 
                tspan=(t[1], t[end]),
                u0=ic,
            ),
            solver;
            saveat = mysaveat,
            kwargs...
    )
end

"""
Returns the a composite loss of the spWindows on ODE data
In this spWindows approach, the Neural Network divides the full interval according to the baseline observations and solves for them separately.
Arguments:
  - `p`: The ODE parameters of the spWindows intervals.
  - `ode_data`: Original Data to be modelled.
  - `weight_matrix`: Weight matrix of size(ode_data) which masks the loss at the appropriate (sliding) places.
  - `tsteps`: Timesteps on which ode_data was calculated.
  - `prob`: ODE problem that the Neural Network attempts to solve.
  - `loss_function`: Any arbitrary function to calculate the loss.
  - `solver`: ODE Solver algorithm.
  - `period_regulizer_multiplier`: The difference between each η_1--η_4 is penalized between periods. This number weights the influence of this penalization.
  - `kwargs`: Additional arguments splatted to the ODE solver. Refer to the
  [Local Sensitivity Analysis](https://diffeq.sciml.ai/dev/analysis/sensitivity/) and
  [Common Solver Arguments](https://diffeq.sciml.ai/dev/basics/common_solver_opts/)
  documentation for more details.
Note:
This implementation heavily builds on the awsome [tutorial](https://diffeqflux.sciml.ai/stable/examples/multiple_shooting/)!
It is readily compatible with Jumps. You can just focus on a single period after declaring the PresetTimeCallback() for the complete TS. Proof of concept:
```julia
showcase_prob = ODEProblem(mydiff, randn(2), (tsteps[1], tsteps[end]), randn(4), callback = PresetTimeCallback(tsteps, update_stressor_level!, save_positions=(true, true)))
temp_prob = remake(showcase_prob, tspan=(tsteps[first(rg)], tsteps[last(rg)]))
plot(solve(testprob3))
plot(solve(temp_prob))
```
# Debugging
```julia
X1,X2,t, W, obs = d[findfirst(x -> x==3872, nonsparse_ids)][1] # 3872 has an interesting missing pattern
x1 = reconstruct(myencoder_ghq, mydecoder_ghq, X1, cpu)[1]
x2  = reconstruct(myencoder_dh, mydecoder_dh, X2, cpu)[1]
ode_data = vcat(x1, x2)
tsteps = t
individual_ranges = split_observation_period_at_first_baseline(obs,W)
p = rand(6)
solver = Tsit5()
prob = ODEProblem(mydiff,[0.0f0, 2.0f0], (0.0f0,maximum(data_raw.pos_obs_grid)), [-0.1, 0.4, -0.2, 0.2]);
```
"""
function spWindows_loss(
    p::AbstractArray, # this needs to be ODEp trained by the ODEnet 
    ode_data::AbstractArray,
    weight_matrix::AbstractArray,
    tsteps::AbstractArray,
    individual_ranges::AbstractArray,
    prob::ODEProblem,
    solver::DiffEqBase.AbstractODEAlgorithm,
    λ_ODEnet::Float64,
    λ_ODEp::Float64,
    λ_sp::Float64,
    kwargs...
)::Float64

    sp_solutions = calc_sp_solutions(tsteps, individual_ranges, p, prob)
    sp_predictions = Array.(sp_solutions)

    # Abort and return infinite loss if one of the integrations failed
    retcodes = [sol.retcode for sol in sp_solutions]
    if any(retcodes .!= :Success)
        return Inf, sp_predictions
    end

    # Calculate spWindow loss
    respondent_loss = 0
    for (i, rg) in enumerate(individual_ranges)
        u = ode_data[:, rg]
        û = sp_predictions[i]
        respondent_loss += mean(abs2, (û .- u)[weight_matrix[:, rg]]) # loss_function(u, û)
    end

    # regularization terms
    ODEnet_reg = λ_ODEnet * sum(x->sum(x.^2), Flux.params(ODEnet))
    ODEp_reg = λ_ODEp * sum(abs, sum.(abs, p)) 

    if size(p,1) > 1 # if condition prevents crashes in case respondents had no second period (sp2)
        sp_reg = λ_sp*sum(abs2, p[1][1:4] - p[2][1:4]) # regularizing the differerence between subperiods
        return respondent_loss += ODEp_reg + ODEnet_reg + sp_reg
    else
        return respondent_loss += ODEp_reg + ODEnet_reg
    end
end

"""
`loss()` is the function which is differentiated. It is necessary to have the ODEp calculation within this function, otherwise the loss cannot be reduced.
# Arguments
- `prob::ODEProblem`: The base problem which receives the ODE parameters from ODEnet.
- `x1::AbstractArray`: 1×T matrix of mental health data.
- `x2::AbstractArray`: 1×T matrix of stressor data.
- `t::AbstractArray`: 1×T matrix of study time in continuous time.
- `W::AbstractArray`: 2×T matrix of weights.
# Debugging
global prob, cur_ranges, sp_regularizer = cur_prob, respondent_ranges, 2.0
"""
function loss_odenet_sp(prob::ODEProblem, x1::AbstractArray, x2::AbstractArray, t::AbstractArray, W::AbstractArray, cur_ranges::AbstractArray, λ_sp::Float64, λ_ODEp::Float64, λ_ODEnet::Float64)
    ODEp_sw = [ODEnet(moduleODEnet.gather_ODE_info(x1[rg], x2[rg], t[rg], W[:,rg])) for rg in cur_ranges]
    return spWindows_loss(ODEp_sw, vcat(x1, x2), W, t, cur_ranges, prob, Tsit5(), λ_sp, λ_ODEp, λ_ODEnet)
end

"""
The `train_node_sp` strongly resembles the `train_node()` function but looks for mnpvisno=1 which indicate new baseline observation in the MARP data. Both create all functions and objects and train/return the ODEnet. No `reconstruct()` is needed since x1 and x2 are sampled from the true ODE directly.
All code within the TRAINING LOOP receives data from a certain respondent. Importantly, the upper part, i.e., everything before `Flux.pullback()` is not differentiated and therefore "object management". It is not possible to update the callback in the loss function or elsewhere (mutating arrays is not supported).
I had also problems providing X1 and X2 (observed matrices) only and apply `reconstruct()` whithin the loss. I guess the reason here is that reconstruct is also differentiated which is currently not necessary. Accordingly, the loss function is as small and clean as possilbe. 
The `loss_odenet_sp()` function furthermore makes use of multiple dispatch. If it sees that auxiliary data is provided, `loss_odenet`()` puts it into the ODEencoder (which concatenates it to the ODE summary statistics).
The `global` part withing the training loop is admitidly ugly! However, I found no other way to provide the changing vector `t` to the `update_stressor_level()` function.
You can initialize a new ODEnet or provide a pretrained which is then further trained (using multiple dispatch).
# Arguments
- `loss`: loss function.
- `data`: data is  Vector{Vector{Tuple. Different data types might be included using multiple dispatch
- `kws`: optional key words to overwrite was had been defined as default in args.jl
# Examples
```julia
julia> train_node_sp(d, base_prob)
Output
```
"""
function train_node_sp(ODEnet, data::AbstractArray, ode_problem::ODEProblem; kws...)

    # debugging with  args = ArgsODEnet(); data = d; i = 1; ode_problem = base_prob;
    args = ArgsODEnet(; kws...)
    args.seed > 0 && Random.seed!(args.seed)
    opt = ADAM(args.η)
    N = size(data,1)
    individual_losses = zeros(N)
    mse = zeros(N)
    period_regulizers = zeros(N)
    epoch_loss = zero(1)
    losses_during_training = zeros(Int(args.epochs/args.verbose_freq), 3)

    !ispath(args.save_path) && mkpath(args.save_path)
    tb_name = string(args.save_path, args.tbidentifier, "spLambda", args.λ_sp , "mylr", args.η, "seed", args.seed, "odep", args.odep_dim, "nhidden", args.hidden_dim, "intensity")

    # logging by TensorBoard.jl, switch only on when serios training is done, e.g. hyperparamter search!
    if args.tblogger
        tblogger = TBLogger(tb_name) 
    end

    local train_loss # training_loss is declared local so it will be available for logging outside the gradient calculation.
    # global stresstest_results = zeros(N, 14); # preallocating result matrix
    ps = Flux.params(ODEnet)

    if args.talk_to_my_repl
        @info "Start Training, total $(args.epochs) epochs with η=$(args.η),  $(args.odep_dim) ODE parameters, and seed $(args.seed)"
    end

    for epoch in 1:args.epochs
        
        learning_index  = args.shuffel_data_per_epoch ? StatsBase.sample(MersenneTwister(args.seed), 1:N, N,replace=false) : 1:N
        
        for i in learning_index
            # These calls to the global environmet below are necessary (afaIk) because of the convoluted algorithm sturcture with cur_affect which looks for a t.
            global z_mh, z_sl, t , W, obs = data[i].predVAE[1,:]' , data[i].predVAE[2,:]', data[i].data.t, data[i].data.Wtrain, data[i].data.mnpvisno # W needs to be loaded globally since it is part of the update_stressor_level!()
            global cur_prob = remake(ode_problem, callback = PresetTimeCallback(t, update_stressor_level!, save_positions=(false,false))) # this might worrk as local as well?
            local respondent_ranges = split_observation_period_at_first_baseline(obs, W)

            train_loss, back = Flux.pullback(ps) do #  code below will be differentiated and needs to be efficient, therefore, all callback stuff happens before!
                loss_odenet_sp(cur_prob, z_mh, z_sl, t, W, respondent_ranges, args.λ_sp, args.λ_ODEp, args.λ_ODEnet)
                end  
                gs = back(1f0)
                Flux.Optimise.update!(opt, ps, gs)
        end
           
        if epoch % args.verbose_freq == 0 || epoch == args.epochs # the loop for calculations at the very end
    
            for i in 1:N
                # W, t, and cur_prob needs to be loaded globally since it is part of the update_stressor_level!() updates.
                global z_mh, z_sl, t , W, obs = data[i].predVAE[1,:]' , data[i].predVAE[2,:]', data[i].data.t, data[i].data.Wtrain, data[i].data.mnpvisno # run on train data only
                # global z_mh, z_sl, t , W, obs = data[i].predVAE[1,:]' , data[i].predVAE[2,:]', data[i].data.t, data[i].data.W, data[i].data.mnpvisno # run on all data
                global cur_prob = remake(ode_problem, callback = PresetTimeCallback(t, update_stressor_level!, save_positions=(false,false))) # this might worrk as local as well?
                
                local respondent_ranges = split_observation_period_at_first_baseline(obs, W)
                local cur_ODEp_sw_reporting = [ODEnet(moduleODEnet.gather_ODE_info(z_mh[rg], z_sl[rg], t[rg], W[:,rg])) for rg in respondent_ranges]
                cur_loss = spWindows_loss(cur_ODEp_sw_reporting, vcat(z_mh, z_sl), W, t, respondent_ranges, cur_prob, Tsit5(), args.λ_sp, args.λ_ODEp, args.λ_ODEnet)
                
                individual_losses[i] = cur_loss
                mse[i] = spWindows_loss(cur_ODEp_sw_reporting, vcat(z_mh, z_sl), W, t, respondent_ranges, cur_prob, Tsit5(), 0.0, 0.0, 0.0) # just return the differences without regularization; handy for hyperparameter search
                
                if size(cur_ODEp_sw_reporting,1) > 1 # in case it sp2 does not exist                    
                    period_regulizers[i] = args.λ_sp*sum(abs2, cur_ODEp_sw_reporting[1][1:4] - cur_ODEp_sw_reporting[2][1:4])
                end
                if epoch == args.epochs # the loop for calculations at the very end
                    sp_solutions = calc_sp_solutions(t, respondent_ranges, cur_ODEp_sw_reporting, cur_prob)
                    if size(cur_ODEp_sw_reporting,1) > 1 # if condition prevents crashes in case respondents had no second period (sp2)
                        data[i].predODE[:,:] = hcat(reduce(hcat, sp_solutions[1].u)[:, 1:end-1], reduce(hcat, sp_solutions[2].u)) # the second lab visit is in there twice, rahter taking IC is a arbitrary decision
                    else
                        data[i].predODE[:,:] = reduce(hcat, sp_solutions[1].u)
                    end       
                end
            end
            epoch_loss = mean(individual_losses)
            period_regulizer_mean = mean(period_regulizers)
            mse_mean = mean(mse)
            if args.talk_to_my_repl
                @info "loss epoch $(epoch)/$(args.epochs): $(round(epoch_loss,digits=4))"
            end
            
            if args.tblogger
                with_logger(tblogger) do
                    @info "individual losses" individual_losses=individual_losses
                    @info "loss" loss=epoch_loss
                    @info "mse" mse=mse_mean
                    @info "period_regulizers" period_regulizers=period_regulizers
                    @info "period_regulizer" period_regulizer=period_regulizer_mean
                    # @info "ODEp" ODEp=ODEp_sw_reporting
                end
            end
            losses_during_training[Int(epoch / args.verbose_freq), :] = [epoch, epoch_loss, mse_mean]        
        end
    end
    if true#args.return_model
        return data, ODEnet, losses_during_training
    end
end
function train_node_sp(data::AbstractArray, ode_problem::ODEProblem; kws...)

    args = ArgsODEnet(; kws...)
    @info "new ODEnet initialized"

    # init ODEnet
    glorot_uniform2(dims...) = (rand(dims...) .- 0.5) .* sqrt(12.0 / sum(dims))
    global ODEnet = Chain(Dense(args.input_dim, args.hidden_dim, relu, init=glorot_uniform2), Dense(args.hidden_dim, args.odep_dim+2, init=glorot_uniform2))
    
    data, ODEnet, losses_during_training = train_node_sp(ODEnet, data, ode_problem; kws...)
    
    return data, ODEnet, losses_during_training
end

# ------
# Plotting
# ------

function plot_subperiods(data, iterator, myODEnet; timestep::Number=0.1, show_plot::Bool=false)

    global mycur_prob, x1, x2, t, W, myrespondent_ranges, myODEp_sw, my_icdist = init_node_objects(data, iterator, base_prob, myencoder_ghq, myencoder_dh, myODEnet)

    
    cur_prob_sp1 = remake(mycur_prob, p=myODEp_sw[1][1:4], u0=u0=myODEp_sw[1][5:6], 
                    tspan=(minimum(t[myrespondent_ranges[1]]), maximum(t[myrespondent_ranges[1]])),
                    callback = PresetTimeCallback(t[myrespondent_ranges[1]], update_stressor_level!, save_positions=(false,false))) 
    
    cur_prob_sp2 = remake(mycur_prob, p=myODEp_sw[2][1:4], u0=u0=myODEp_sw[2][5:6], 
                    tspan=(minimum(t[myrespondent_ranges[2]]), maximum(t[myrespondent_ranges[2]])),
                    callback = PresetTimeCallback(t[myrespondent_ranges[2]], update_stressor_level!, save_positions=(false,false))) 

    cur_sol_sp1 = solve(cur_prob_sp1, saveat = timestep)
    cur_sol_sp2 = solve(cur_prob_sp2, saveat = timestep)

    myplot = plot(cur_sol_sp1.t, reduce(hcat, cur_sol_sp1.u)', c=[:blue :red], lab=["P (sp1)" "E (sp1)"], ylim=(-1, 2), lw = 3, legend = false)
    plot!(myplot, cur_sol_sp2.t, reduce(hcat, cur_sol_sp2.u)', c=[:blue :red], lw = 3)
    scatter!(myplot, t, vec(x1), c=[:blue], lab="P (VAE)", ms = 6)
    scatter!(myplot, t, vec(x2), c=[:red],  lab="E (VAE)", ms = 6)
    
    if show_plot
        display(myplot)
    end

    return(myplot)
end

export train_node_sp, plot_subperiods
# end # end of the module

# ------
# LEGACY
# ------

# if i == 1 && args.plot_solutions_while_learning
#     # ODEp = [ODEnet(moduleODEnet.gather_ODE_info(z_mh[rg], z_sl[rg], t[rg], W[:,rg])) for rg in respondent_ranges]
#     period_solutions = calc_sp_solutions(t, respondent_ranges, cur_ODEp_sw_reporting, ode_problem)
#     @suppress begin # supress warnings 
#         visual_callback(z_mh, z_sl, t, W, period_solutions, respondent_ranges)
#     end
# end

# """
# This function takes the data, the VAEs, and the ODEnet and returns the components mapped into the latent space and the ODE problem.
# # Debugging
# ```julia
# julia> data, i, ode_problem, ghq_encoder, dh_encoder, odeNet = d, 2, base_prob, myencoder_ghq, myencoder_dh, ODEnet
# ```
# """
# function init_node_objects(data, i::Int, ode_problem::ODEProblem, ghq_encoder, dh_encoder, odeNet)

#     global X1, X2, t, W, obs = data[i].data.Xmh, data[i].data.Xsl, data[i].data.t, data[i].data.W, data[i].data.mnpvisno
#     global x1 = moduleODEnet.reconstruct(ghq_encoder, mydecoder_ghq, X1, cpu)[1]
#     global x2 = moduleODEnet.reconstruct(dh_encoder, mydecoder_dh, X2, cpu)[1]
      
#     respondent_ranges = split_observation_period_at_first_baseline(obs, W)
#     ODEp_sw = [odeNet(gather_ODE_info(x1[rg], x2[rg], t[rg], W[:,rg])) for rg in respondent_ranges]
    
#     global cur_prob = remake(ode_problem, callback = PresetTimeCallback(t, update_stressor_level!, save_positions=(false,false))) 
    
#     μ, logσ = ghq_encoder(X1[:,1])
#     ic_dist = Normal(μ[1], exp(logσ[1]))
#     sp_solutions = calc_sp_solutions(t, respondent_ranges, ODEp_sw, cur_prob)

#     trajectory_predictions = hcat(reduce(hcat, sp_solutions[1].u)[:, 1:end-1], reduce(hcat, sp_solutions[2].u)) # the lab visit is in there twice, rahter taking IC is a arbitrary decision

#     return cur_prob, x1, x2, t, W, respondent_ranges, ODEp_sw, ic_dist, sp_solutions, trajectory_predictions
# end





                #     stresstest_results[i, 1:8] = vcat(nonsparse_ids[i], cur_loss, cur_ODEp_sw_reporting[1])

                #     if i ∈ indicies_meeting_minimal_sp_requirements
                #         stresstest_results[i, 9:end] = cur_ODEp_sw_reporting[2]
                #     end
                    
                #     if args.finally_plot_all_solutions
                        
                #             post_parameters = ODEencoder(z_mh, z_sl, t, W)
                #             finegrained_solution = solve(p=post_parameters[1:4],cur_prob, Tsit5(), u0=post_parameters[5:6], save_positions=(true, true), 
                #                                     callback = PresetTimeCallback(t, update_stressor_level!, save_positions=(true,true)), sensealg=ReverseDiffAdjoint(), saveat = 0.08)#[1:2,:] 
        
                #             @suppress begin # supress warnings 
                #                 my_plot = plot(finegrained_solution, title="solution for i: $i", ylim=(-1, 2.5), c=[:darkblue :darkred], lab=["GHQ (ODE)" "DH (ODE)"])
                #                 scatter!(my_plot, t, vec(z_mh), c=[:blue], lab="GHQ (VAE)")
                #                 scatter!(my_plot, t, vec(z_sl), c=[:red],  lab="DH (VAE)")
                #                 display(my_plot)
                #             end
                #             savefig("figures/preliminary_figures/lambda_comparison/id$i.png")
                #     end
                #     mydf = DataFrame(stresstest_results, :auto)
                #     rename!(mydf, [:id, :loss, :drift_ghq_sp1, :dh_on_ghq_sp1, :drift_dh_sp1, :ghq_on_dh_sp1, :intercept_ghq_sp1, :intercept_dh_sp1, :drift_ghq_sp2, :dh_on_ghq_sp2, :drift_dh_sp2, :ghq_on_dh_sp2, :intercept_ghq_sp2, :intercept_dh_sp2])
                #     CSV.write(string(tb_name,".csv"), mydf)
                # end

# Animate training
# visual_callback = function (x1, x2, t, W, solutions, ranges)
#     preds = Array.(solutions)
#   plt = scatter(t[W[1,:] .== 1], x1[W[1,:] .== 1], label = "GHQ (VAE)", markersize = 21, c=[:blue]) # plot the original data
#     for (i, rg) in enumerate(ranges) # plot the different predictions for the ranges
#         plot!(plt, t[rg], preds[i][1,:], markershape=:circle, c=[:darkblue], label="Baseline Period $(i)", lw=6)
#     end 
#     frame(anim)
#     display(plot(plt))
# end

# # this function samples the ICs, this was an idea Harald had once.
# function calc_sp_differences(data;  nIcSamples::Int = 10, time_grid = 0.5:0.5:3)
    
#     my_results = zeros(size(indicies_meeting_minimal_sp_requirements, 1), 11+size(time_grid,1)); 

#     for (i, id) in enumerate(indicies_meeting_minimal_sp_requirements)
        
#         mycur_prob, myx1, myx2, myt, myW, myrespondent_ranges, myODEp_sw, my_icdist, ode_trajectories = init_node_objects(data, id, base_prob, myencoder_ghq, myencoder_dh, ODEnet)
#         my_ics = rand(MersenneTwister(0), my_icdist, nIcSamples)
#         my_differences = zeros(nIcSamples, size(time_grid,1))

#         for nIcSample in 1:nIcSamples
#             plot(solve(p=myODEp_sw[1][1:4], remake(base_prob, u0=[my_ics[nIcSample]; 2.0])), vars=(1))
#             plot!(solve(p=myODEp_sw[2][1:4], remake(base_prob, u0=[my_ics[nIcSample]; 2.0])), vars=(1))
            
#             my_differences[nIcSample, :] = diff(hcat(
#                 Array(solve(p=myODEp_sw[1][1:4], remake(base_prob, u0=[my_ics[nIcSample]; 2.0]), saveat=time_grid))[1,:], 
#                 Array(solve(p=myODEp_sw[2][1:4], remake(base_prob, u0=[my_ics[nIcSample]; 2.0]), saveat=time_grid))[1,:]), 
#                 dims = 2)
#         end
#         # 
#         # Symbol.(["mnppid", "odep11", "odep12", "odep13", "odep14", "odep21", "odep22", "odep23", "odep24", "ICmu", "ICsigma", "ptp1", "ptp2", "ptp3", "ptp4", "ptp5", "ptp6"])
#         my_results[i,:] = vcat(nonsparse_ids[id], 
#                                 myODEp_sw[1][1:4],
#                                 myODEp_sw[2][1:4],
#                                 my_icdist.μ,
#                                 my_icdist.σ,
#                                 mean(my_differences, dims = 1)[1:end])
#     end
#     return my_results
# end

# """
# Get ranges that partition data of length `datasize` in groups of `groupsize` observations.
# If the data isn't perfectly dividable by `groupsize`, the last group contains
# the reminding observations.
# ```julia
# group_ranges(datasize, groupsize)
# ```
# Arguments:
# - `datasize`: amount of data points to be partitioned
# - `groupsize`: maximum amount of observations in each group
# Example:
# ```julia-repl
# julia> group_ranges(10, 5)
# 3-element Vector{UnitRange{Int64}}:
#  1:5
#  5:9
#  9:10
# ```
# """
# function group_ranges(datasize::Integer, groupsize::Integer)
#     2 <= groupsize <= datasize || throw(
#         DomainError(
#             groupsize,
#             "datasize must be positive and groupsize must to be within [2, datasize]",
#         ),
#     )
#     return [i:min(datasize, i + groupsize - 1) for i in 1:groupsize-1:datasize-1]
# end
# function spWindows(datasize::Integer, groupsize::Integer)
#     2 <= groupsize <= datasize || throw(
#         DomainError(
#             groupsize,
#             "datasize must be positive and groupsize must to be within [2, datasize]",
#         ),
#     )
#     return [i:min(datasize, i + groupsize - 1) for i in 1:(datasize-groupsize+1)]
# end
# # spWindows(15, 4)