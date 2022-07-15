module moduleVAE

using Distributions, Flux, BSON, Plots, Random, MLUtils, TensorBoardLogger
using Logging: with_logger
include("../src/ArgsVAE.jl")

"""
kl_q_p_VAE is the closed form solution for the Kullback–Leibler divergence for normally distribted distributions.
# Arguments
- `μ`: The mean of the distribution.
- `logs`: The log(σ), i.e., the logarithm of the variance.
# Examples
```julia
julia> kl_q_p_VAE(1, 0.8)
1.6765162121975574
```
"""
kl_q_p_ODE(μ, logs) = 0.5 * sum(exp.(2 .* logs) + μ.^2 .- 1 .- (2 .* logs), dims=1)
kl_q_p_VAE(μ, logs) = 0.5 * sum(exp.(2 .* logs) + μ.^2 .- 1 .- (2 .* logs))

struct Encoder
    linear
    μ
    logσ
    Encoder(input_dim, latent_dim, hidden_dim, device) = new(
        Dense(input_dim, hidden_dim, tanh) |> device,   # linear
        Dense(hidden_dim, latent_dim) |> device,        # μ
        Dense(hidden_dim, latent_dim) |> device,        # logσ
    )
end

function (encoder::Encoder)(x)
    h = encoder.linear(x)
    encoder.μ(h), encoder.logσ(h)
end

Decoder(input_dim, latent_dim, hidden_dim, device) = Chain(
    Dense(latent_dim, hidden_dim, tanh),
    Dense(hidden_dim, input_dim, relu) # this activation function is an important choice! relu bc Poisson(), other options available
) |> device

function reconstruct(encoder, decoder, x, device; my_seed = 123)
    μ, logσ = encoder(x)
    z = μ + device(randn(MersenneTwister(my_seed), Float32, size(logσ))) .* exp.(logσ) # reparametrization trick, in order to make it reproducible, you need to provide a seed to the random noise
    μ, logσ, decoder(z)
end

"""
The Poisson loss function which gets fliped for log p_{x|z}.
# λ: Poisson parameter >= 0
# Notes
# You might investigate performance with @time, not necessary to use Poisson() here...
# Examples
```julia
julia> poisson_loss(Poisson(1.8), 2.0) < poisson_loss(Poisson(1.2), 2.0)
-0.7752714710650175
julia> sum(poisson_loss.(Poisson.(mydec([1.2, 1.4])), ghqm[:,1]))
```
"""
poisson_loss(p::Poisson{Float32}, y::Number) = (p.λ - y * log(p.λ + eps()))

"""
vae_loss() is the loss function of the variational autoencoder. It combines the inverse of the log-likelihood with the Kullback–Leibler divergence and an additional regularization term for penalizing the weights of the decoder.

# Arguments
- `encoder`: VAE encoder.
- `decoder`: VAE decoder.
- `λ_enc`: Penality term of weight regularization for the encoder.
- `λ_dec`: Penality term of weight regularization for the decoder.
- `x`: The p x batch_size matrix of observations.
- `device`: Either cpu org gpu.
- `myseed`: To increase reporducibility, the seed must be specified for sample the same noise.
```
"""
Flux.Zygote.@nograd Flux.params # https://discourse.julialang.org/t/mutating-arrays-is-not-supported-error-when-running-fluxml-vae-mnist-jl/59890/4
function vae_loss(encoder, decoder, λ_enc, λ_dec, x, device, myseed)
    μ, logσ, reconstructed_x = reconstruct(encoder, decoder, x, device; my_seed = myseed)
    len = size(x)[end]
    
    kl_q_p = 0.5f0 * sum(@. (exp(2f0 * logσ) + μ^2 -1f0 - 2f0 * logσ)) / len # KL-divergence
    logp_x_z = -sum(poisson_loss.(Poisson.(reconstructed_x), x)) / len # log p(x|z)
    reg_dec = λ_enc * sum(x->sum(x.^2), Flux.params(decoder)) # regularization
    reg_enc = λ_dec * sum(x->sum(x.^2), Flux.params(encoder.linear))
    
    return -logp_x_z + kl_q_p + reg_dec + reg_enc
end

"""
The `train_vae()` function initializes the encoder and decoder and trains their parameters or takes a pretrained decoder/encoder and trains them on the data.
    It is heavily inspired be the [model zoo](https://github.com/FluxML/model-zoo/tree/master/vision/vae_mnist) implementation.
# Arguments
- `n::Integer`: the number of elements to compute.
# Examples
```julia
julia> FunctionName
Output
```
"""
function train_vae(data::AbstractMatrix; kws...)
    # load hyperparamters
    args = ArgsVAE(; kws...) # debugging with  args = ArgsVAE()
    args.seed > 0 && Random.seed!(args.seed)

    # GPU config
    if args.cuda # && has_cuda_gpu() # package installation of CUDAapi not successful, therefore abandoned for the moment
        device = gpu
    else
        device = cpu
    end

    # load data as mini batches of size args.batch_size
    loader = MLUtils.DataLoader(data, batchsize=args.batch_size, shuffle=true)
    
    # initialize encoder and decoder; supplying pretrained en/decoder would be nice!
    encoder = Encoder(args.input_dim, args.latent_dim, args.hidden_dim, device)
    decoder = Decoder(args.input_dim, args.latent_dim, args.hidden_dim, device)

    # ADAM optimizer
    opt = ADAM(args.η)
    
    # parameters
    ps = Flux.params(encoder.linear, encoder.μ, encoder.logσ, decoder)

    !ispath(args.save_path) && mkpath(args.save_path)

    if args.tblogger
        tblogger = TBLogger(string(args.save_path, args.tbidentifier, "mylr", args.η, "seed", args.seed, "batchs", args.batch_size)) 
    end
    
    if args.train
        if args.talk_to_my_repl
            @info "Start Training, total $(args.epochs) epochs with η=$(args.η) and batch size $(args.batch_size)"
        end

        for epoch = 1:args.epochs
            for x in loader 
                batch_loss, back = Flux.pullback(ps) do
                    vae_loss(encoder, decoder, args.λ_enc, args.λ_dec, x |> device, device, args.seed)
                    end
                grad = back(1f0)
                Flux.Optimise.update!(opt, ps, grad)
            end
            if epoch % args.verbose_freq == 0
                epoch_loss = vae_loss(encoder, decoder, args.λ_enc, args.λ_dec, loader.data |> cpu, cpu, args.seed)
                
                if args.talk_to_my_repl
                    @info "loss epoch $(epoch)/$(args.epochs): $(round(epoch_loss,digits=2))"
                end
                if args.tblogger
                    with_logger(tblogger) do
                        @info "train" loss=epoch_loss
                    end
                end
            end
        end
        # save and/or return model
        if args.save_model
        model_path = joinpath(args.save_path, args.save_model_object)
            let encoder = cpu(encoder), decoder = cpu(decoder), args=struct2dict(args)
                BSON.@save model_path encoder decoder args
                @info "Model saved: $(model_path)"
            end
        end
    end
    return encoder, decoder
end

# # Function used in the Notebook and the main script
function summary_plot_vae(ghqm, dhm, myencoder_ghq, mydecoder_ghq, myencoder_dh, mydecoder_dh; mytitle="Overview of VAE training")

    z_ghq, sigma_ghq, reconstructed_z_ghq = reconstruct(myencoder_ghq, mydecoder_ghq, ghqm, cpu);
    z_dh, sigma_dh, reconstructed_z_dh = reconstruct(myencoder_dh, mydecoder_dh, dhm, cpu);
    
    r1 = cor(vec(z_ghq), sum(ghqm', dims=2))
    r2 = cor(vec(z_dh), sum(dhm', dims=2))

    p1 = Plots.histogram(vec(z_ghq), xaxis=string("Distribution of Zs (mh)", "overall mean = ", round(mean(vec(z_ghq)), digits=3)), legend = false, title=mytitle)

    p4 = Plots.histogram(vec(z_dh), xaxis=string("Distribution of Zs (sl)", "overall mean = ", round(mean(vec(z_dh)), digits=3)), legend = false)

    p2 = Plots.scatter(vec(z_ghq), vec(sum(ghqm, dims=1)), legend = false,
        xaxis=string("latens scores; r = ", round(r1.parent[1], digits=2)),
        yaxis=string("sum scores (ghq)"))

    p3 = Plots.scatter(vec(z_dh), sum(dhm', dims=2), legend = false, 
        xaxis=string("latens scores; r = ", round(r2.parent[1], digits=2)),
        yaxis=string("sum scores (dh)"))

    # display(plot(p2, p3, p4, p5, layout=grid(2, 2, heights=[0.7,0.3], widths=[0.6, 0.4]), size=[1050, 1050]))
    display(plot(p1, p2, p3, p4, layout=grid(2, 2), size=[1050, 1050]))
end

export reconstruct, summary_plot_vae, train_vae

end # end of module