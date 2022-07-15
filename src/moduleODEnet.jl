module moduleODEnet

using DifferentialEquations, DiffEqSensitivity, DataFrames, TensorBoardLogger, Flux, Suppressor, Random, StatsBase
using Statistics: mean
using Logging: with_logger
include("../src/ArgsODEnet.jl")

struct Data
    id::Int
    Xmh::Matrix{Float64} # n_t × p_ghq matrix of ghq items
    Xsl::Matrix{Float64}  # n_t × p_dh  matrix of dh  items
    t::Vector{Float64} # continuous time
    mnpvisno::Vector{Int64} # discrete time but with mnpvisno == 1 for second lab visit
    W::BitMatrix # the 2 x n_t matrix with all obsvervations (if no out-of-sample prediction is intended)
    Wtrain::BitMatrix # the 2 x n_t matrix of weights for training
    Wtest::BitMatrix # the 2 x n_t matrix indicating the last obs (for evaluation)
end
mutable struct subject
    data::Data
    predVAE::Matrix{Float64} # placeholder for VAE mappings
    predODE::Matrix{Float64} # placeholder for ODE predictions
end

"""
A non-generic function tailored to the MARP data set.
# Arguments
- `id`: the number of elements to compute.
- `data`: A dataframe in the wide format with the items starting with "ghq" and "hasa".
# Examples
```julia
julia> extract_subject_from_dataframe(3758, data_raw)
(3758, [2.0 1.0 … 2.0 1.0; 2.0 1.0 … 1.0 1.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0], [1.0 0.0 … 0.0 0.0; 1.0 0.0 … 1.0 0.0; … ; 1.0 0.0 … 0.0 0.0; 1.0 0.0 … 0.0 0.0], [0.0, 0.48, 1.49, 2.48, 3.49, 4.48, 5.46, 6.46, 6.98, 8.12  …  10.12, 11.11, 12.11, 13.11, 14.11, 15.1, 16.1, 17.09, 18.09, 19.09], [1, 2, 3, 4, 5, 6, 7, 8, 1, 10  …  12, 13, 14, 15, 16, 17, 18, 19, 20, 21], Bool[1 1 … 1 1; 0 1 … 1 1])
```
"""
function extract_subject_from_dataframe(id::Int, data::DataFrame, replaceMissings::Bool=true, dropMissings::Bool=false, missingReplacement::Number=1.0)
    id_subset = filter(x-> x[:mnppid]  == id, data)
    id_subset.pos_obs_grid = round.(id_subset.pos_obs_grid .- minimum(id_subset.pos_obs_grid), digits=3) # every id_subset starts with pos_obs_grid = 0.0; regardless of first obs was complete or not
   
    if replaceMissings
        id_subset = deepcopy(coalesce.(id_subset, missingReplacement))
    end
    if dropMissings
        id_subset = dropmissing!(id_subset)
    end

    Xmh = Array{Float64,2}(id_subset[:, names(id_subset, r"^ghq")])'
    Xsl = Array{Float64,2}(id_subset[:, names(id_subset, r"^hasa")])'
    t = id_subset.pos_obs_grid
    mnpvisno = id_subset.mnpvisno
    W = hcat(id_subset.wghq, id_subset.wdh)' .== 1.0
    return (id, Xmh, Xsl, t, mnpvisno, W)
end
"""
A conSTUCTor to bring the data from a DataFrame to the right strucutre.
# Arguments
- `id`: the number of elements to compute.
- `Xmh`: A matrix of mental health items.
- `Xsl`: A matrix of stressor load items.
- `t`: A vector of continuous time.
- `visno`: A vector of discrete time.
# Examples
```julia
julia> conSTRUCTor(extract_subject_from_dataframe(3758, data_raw)...)
subject(Main.moduleODEnet.Data(3758, [2.0 1.0 … 2.0 1.0; 2.0 1.0 … 1.0 1.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0], [1.0 0.0 … 0.0 0.0; 1.0 0.0 … 1.0 0.0; … ; 1.0 0.0 … 0.0 0.0; 1.0 0.0 … 0.0 0.0], [0.0, 0.48, 1.49, 2.48, 3.49, 4.48, 5.46, 6.46, 6.98, 8.12  …  10.12, 11.11, 12.11, 13.11, 14.11, 15.1, 16.1, 17.09, 18.09, 19.09], [1, 2, 3, 4, 5, 6, 7, 8, 1, 10  …  12, 13, 14, 15, 16, 17, 18, 19, 20, 21], Bool[1 1 … 1 1; 0 1 … 1 1], Bool[1 1 … 1 0; 0 1 … 1 0], Bool[0 0 … 0 1; 0 0 … 0 1]), [1.2757691969704335 0.44992354186501565 … 0.8540318849197894 0.4180865681108882; 1.9808694704091858 1.244273976266171 … 0.8042047144188857 1.0816387365631939], [0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0])
```
"""
function conSTRUCTor(id, Xmh, Xsl, t, mnpvisno, W)
    
    Wtrain = deepcopy(W)
    Wtrain[1, findlast(W[1,:] .== 1)] = false
    Wtrain[2, findlast(W[2,:] .== 1)] = false
    
    Wtest = zeros(size(W))
    Wtest[1, findlast(W[1,:] .== 1)] = true
    Wtest[2, findlast(W[2,:] .== 1)] = true

    return moduleODEnet.subject(
        moduleODEnet.Data(
            id, 
            Xmh,
            Xsl,
            t,
            mnpvisno,
            W,
            Wtrain,
            Wtest),
        zeros(size(W)),
        zeros(size(W))
    )
end

function reconstruct(encoder, decoder, x, device; my_seed = 123)
    μ, logσ = encoder(x)
    z = μ + device(randn(MersenneTwister(my_seed), Float32, size(logσ))) .* exp.(logσ) # reparametrization trick, in order to make it reproducible, you need to provide a seed to the random noise
    μ, logσ, decoder(z)
end

"""
stepintegrate() calculates the piecewise constant step functions to approximate the integral of the individual trajectories.
# Arguments
- `t`: the time vector.
- `x`: the vector of latent measures.
```
"""
stepintegrate(t, x) = sum(diff(t) .* x[2:end]) 
stepsqintegrate(t, x) = sum(diff(t) .* x[2:end].^2) 
stepscintegrate(t, x) = sum(diff(t) .* x[2:end].^3) 
stepsaintegrate(t, x) = sum(abs.(diff(t) .* x[2:end])) 

"""
`gather_ODE_info()` computes the summary statistics which are the inputs of the ODEnet. It creats t1, t2, x1, anc x2 inside the function. This might not be the most efficient thing to do, however, it eases up the data structure/handling a bit.
It does apply the weights to the vectors, no need for taking care any further.
# Arguments
- `x1`: vector of the first group of variables (observations) in the latent space, usually mental health.
- `x1`: vector of the first group of variables (observations) in the latent space, usually mental health.
# Examples
```julia
julia> gather_ODE_info()
Output
# Debugging
```julia
julia> X1, X2, t, w, obs = d[4][1]
julia> x1, x2 = reconstruct(myencoder_ghq, mydecoder_ghq, X1, cpu)[1], reconstruct(myencoder_dh, mydecoder_dh, X2, cpu)[1]
```
"""
function gather_ODE_info(x1, x2, t, w; denominator::Int=10)  

    t1 = t[w[1,:] .== 1]
    t2 = t[w[2,:] .== 1]
    x1 = x1[w[1,:] .== 1]
    x2 = x2[w[2,:] .== 1]
    tmax = t[end]
    
    [x1[1],
        x1[1] - x1[end],
        stepintegrate(t1, x1) / tmax,
        stepsaintegrate(t1, x1) / tmax,
        stepsqintegrate(t1, x1) / tmax,
        # mean(autocor(vec(x1))) * 100, # does not work with Zygote, throws `mutating arrays is not supported`...
        # mean(autocor(vec(x2))) * 100,
        x2[1],
        x2[1] - x2[end],
        x1[1] - x2[1],
        x1[end] - x2[end],
        x1[1] - x2[end],
        x1[end] - x2[1],
        stepintegrate(t2, x2) / tmax,
        stepsaintegrate(t2, x2) / tmax,
        stepsqintegrate(t2, x2) / tmax
    ] / denominator
end

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
`ODEencoder` simply maps its inputs to the `ODEnet` and `gather_ODE_info()`. It is defined globally to ease up life, especially when it comes to plotting.
"""
ODEencoder(x1, x2, t, W) = ODEnet(gather_ODE_info(x1, x2, t, W))
"""
`enode` is just a tiny wrapper around `solve()`. It is defined globally to ease up life, especially when it comes to plotting.
"""
enode(x1,x2,current_p, t, current_problem) = solve(p=current_p[1:4],current_problem, Tsit5(), u0=current_p[5:6], saveat = t, sensealg=ReverseDiffAdjoint())    

"""
loss_odenet(t,x1,x2) uses multiple dispatch, it adds aux data to the `gather_ODE_info()` data so the rest can stay largely the same (you currently still need to adapt the input dimensions).
# Examples
```julia
julia> loss_odenet(ODEencoder, enode, prob, d[1])
39.41376452468255
```
# Debugging
```julia
X1,X2,t, W, obs = d[findfirst(x -> x==3872, nonsparse_ids)][1] # 3872 has an interesting missing pattern
current_p=[0.0, 0.0, 0.0, 0.0, 1.0, 1.2]
enode(x1, x2, curp, t, base_prob)
cur_prob = remake(base_prob, callback = PresetTimeCallback(t, update_stressor_level!, save_positions=(false,false))) # try local!
solve(p=current_p[1:4],cur_prob, Tsit5(), u0=current_p[5:6], saveat = t, save_positions=(false,false), sensealg=ReverseDiffAdjoint())    
solve(p=current_p[1:4],cur_prob, Tsit5(), u0=current_p[5:6], saveat = t, sensealg=ReverseDiffAdjoint())    
```
"""
function loss_odenet(prob::ODEProblem, x1::AbstractMatrix{Float64}, x2::AbstractMatrix{Float64}, subj::subject)
    curp = ODEencoder(x1, x2, subj.data.t, subj.data.Wtrain)
    pred = Array(enode(x1,x2, curp, subj.data.t, prob))
    return Statistics.mean(abs2, (pred .- vcat(x1,x2))[subj.data.Wtrain])
end

"""
`train_node()` creates all functions and objects and trains/returns the ODEnet. 
All code within the TRAINING LOOP receives data from a certain respondent. Importantly, the upper part, i.e., everything before `Flux.pullback()` it is not differentiated and therefore "object management". It is not possible to update the callback in the loss function or elsewhere (mutating arrays is not supported).
I had also problems providing X1 and X2 (observed matrices) only and apply `reconstruct()` whithin the loss. I guess the reason here is that reconstruct is also differentiated which is currently not necessary. Accordingly, the loss function is as small and clean as possilbe. 
The `loss_odenet()` function furthermore makes use of multiple dispatch. If it sees that auxiliary data is provided, `loss_odenet`()` puts it into the ODEencoder (which concatenates it to the ODE summary statistics).
The `global` part withing the training loop is admitidly ugly! However, I found no other way to provide the changing vector `t` to the `update_stressor_level()` function.
# Arguments
- `data`: data is  Vector{Vector{Tuple. Different data types might be included using multiple dispatch
- `kws`: optional key words to overwrite was had been defined as default in args.jl
# Examples
```julia
julia> train_node()
Output
```
"""
function train_node(data::Vector{subject}, ode_problem::ODEProblem; kws...)

    args = ArgsODEnet(; kws...) # debugging with  args = ArgsODEnet()
    args.seed > 0 && Random.seed!(args.seed)
    opt = Flux.ADAM(args.η)
    n = size(data,1)
    individual_losses = zeros(n)
    epoch_loss = zero(1)

    !ispath(args.save_path) && mkpath(args.save_path)

    # logging by TensorBoard.jl, switch only on when serios training is done, e.g. hyperparamter search!
    if args.tblogger
        tblogger = TBLogger(string(args.save_path, args.tbidentifier, "mylr", args.η, "seed", args.seed, "odep", args.odep_dim), tb_overwrite) 
    end

    # init ODEnet
    glorot_uniform2(dims...) = (rand(Float32, dims...) .- 0.5f0) .* sqrt(12.0f0 / sum(dims))
    global  ODEnet = Chain(Dense(args.input_dim, args.hidden_dim, relu, init=glorot_uniform2), Dense(args.hidden_dim, args.odep_dim+2, init=glorot_uniform2))
    local train_loss # training_loss is declared local so it will be available for logging outside the gradient calculation.
    ps = Flux.params(ODEnet)

    if args.talk_to_my_repl
        @info "Start Training, total $(args.epochs) epochs with η=$(args.η),  $(args.odep_dim) ODE parameters, and seed $(args.seed)"
    end

    for epoch in 1:args.epochs

        learning_index  = args.shuffel_data_per_epoch ? StatsBase.sample(MersenneTwister(args.seed), 1:N, N,replace=false) : 1:N

        for i in learning_index

            global z_mh, z_sl, t , W = data[i].predVAE[1,:]' , data[i].predVAE[2,:]', data[i].data.t, data[i].data.W # W needs to be loaded globally since it is part of the update_stressor_level!()
            global cur_prob = remake(ode_problem, callback = PresetTimeCallback(t, update_stressor_level!, save_positions=(false,false))) # this might worrk as local as well?
            
            train_loss, back = Flux.pullback(ps) do #  code below will be differentiated and needs to be efficient, therefore, all callback stuff happens before!
                    loss_odenet(cur_prob, z_mh, z_sl, data[i])
                end              
                gs = back(1f0)
                Flux.Optimise.update!(opt, ps, gs)
            end
           
            if epoch % args.verbose_freq == 0 
                for i in 1:n
                    global z_mh, z_sl, t , W = data[i].predVAE[1,:]' , data[i].predVAE[2,:]', data[i].data.t, data[i].data.W # W needs to be loaded globally since it is part of the update_stressor_level!()
                    global cur_prob = remake(ode_problem, callback = PresetTimeCallback(t, update_stressor_level!, save_positions=(false,false))) # this might worrk as local as well?
            
                    individual_losses[i] = loss_odenet(cur_prob, z_mh, z_sl, data[i])

                    if i == 1 && args.plot_solutions_while_learning

                        post_parameters = ODEencoder(z_mh, z_sl, t, W)
                        post_solution = solve(p=post_parameters[1:4],cur_prob, Tsit5(), u0=post_parameters[5:6], callback = PresetTimeCallback(t, update_stressor_level!, save_positions=(true,true)), sensealg=ReverseDiffAdjoint(), kwargshandle=KeywordArgError)

                        @suppress begin # supress warnings 
                            my_plot = plot(post_solution, title="my_title", ylim=(-3, 3), c=[:darkblue :darkred], lab=["GHQ (ODE)" "DH (ODE)"])
                            scatter!(my_plot, t, vec(x1), c=[:blue], lab="GHQ (VAE)")
                            scatter!(my_plot, t, vec(x2), c=[:red],  lab="DH (VAE)")
                            display(my_plot)
                        end
                    end

                    if epoch == args.epochs && args.finally_plot_all_solutions

                        post_parameters = ODEencoder(z_mh, z_sl, t, W)
                        post_solution = solve(p=post_parameters[1:4],cur_prob, Tsit5(), u0=post_parameters[5:6], save_positions=(true, true), callback = PresetTimeCallback(t, update_stressor_level!, save_positions=(true,true)), sensealg=ReverseDiffAdjoint())#[1:2,:] 

                        @suppress begin # supress warnings 
                            my_plot = plot(post_solution, title="my_title", ylim=(-1, 3), c=[:darkblue :darkred], lab=["GHQ (ODE)" "DH (ODE)"])
                            scatter!(my_plot, t, vec(z_mh), c=[:blue], lab="GHQ (VAE)")
                            scatter!(my_plot, t, vec(z_sl), c=[:red],  lab="DH (VAE)")
                            display(my_plot)
                        end
                    end
                end
                epoch_loss = Statistics.mean(individual_losses)
                if args.talk_to_my_repl
                    @info "loss epoch $(epoch)/$(args.epochs): $(round(epoch_loss,digits=2))"
                end
                
                if args.tblogger
                    # a more informative logging might look like this: println(string("L_z1 = ", round(loss_smoothz1, digits=1), "; L_z2 = ", round(loss_smoothz2, digits=1), "; Regu ( gets*", regularization_multi, ") = ", round(loss_regularization, digits=1)))
                    # this requires to write a new loss function which returns the details before summing it up, or use a if switch inside the existing one.
                    with_logger(tblogger) do
                        @info "individual losses" individual_losses=individual_losses
                        @info "mse" mse=epoch_loss
                    end
                end
            end
    end
    if args.return_model
        return ODEnet
    end
end

# DEFINED SEVERAL TIMES; MIGHT GO AWAY ENTIRELY
function plot_node_solution(X1, X2, t, W, obs, ode_problem, ylimits::Tuple=(-6, 6))    
    
    cur_prob = remake(ode_problem, callback = PresetTimeCallback(t, update_stressor_level!, save_positions=(false,false))) 
    
    post_parameters = ODEencoder(x1, x2, t, W)
    post_solution = solve(p=post_parameters[1:4],cur_prob, Tsit5(), u0=post_parameters[5:6], save_positions=(true, true), callback = PresetTimeCallback(t, update_stressor_level!, save_positions=(true,true)), sensealg=ReverseDiffAdjoint())
    
    @suppress begin # supress warnings 
        my_plot = plot(post_solution, title="my_title", ylim=(-3, 3), c=[:darkblue :darkred], lab=["GHQ (ODE)" "DH (ODE)"])
        scatter!(my_plot, t, vec(x1), c=[:blue], lab="GHQ (VAE)")
        scatter!(my_plot, t, vec(x2), c=[:red],  lab="DH (VAE)")
        display(my_plot)
    end
end

export conSTRUCTor, train_node, Data, subject
end # this is the end of the module
