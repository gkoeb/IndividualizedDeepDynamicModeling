
include("../src/moduleODEnet_sp.jl")
include("../src/moduleODEnet.jl")

import .moduleODEnet_sp: calc_solution
import .moduleODEnet: Data
import .moduleODEnet: subject
import .moduleODEnet: conSTRUCTor

using DifferentialEquations, StatsBase, Distributions, Plots, StableRNGs

"""
`sample_events()` sample the timing and strength of events that happen to the simulated respondents. To simulate the occurrence of adverse events, we need to sample two quantities: timing and strength. 
The number of the events is governed by the intensity of the Poisson process. For sampling time, we make tha assumption that the conditional intensity is constant and independent of the event history. Accordingly, we sample the the timing of the events from a Poisson process. Technically, simulating the timing of the events involves simulating both the number and location of the events. While the number of the events is governed by the intensity, the location can be sampled uniformly along the time span (with our assumptions). Another assumption is that the timing and strength are independent. Truncation is applied to the event number [1,20] and strength [0.5,5]; this ensures at least one event per observation period and moderate to strong event strengths.
# Arguments
- `intensity`: Intensity (λ) of the Poisson process
- `IGammaAlpha`: α of Inverse Gamma distribution (shape parameter)
- `IGammaTheta`: θ of Inverse Gamma distribution (scale parameter)
- `tmin`: minimum of t, usually 0
- `tmax`: maximum of t, default 20
- `truncate_lower:` Truncate the samples of the Inverse Gamma distirubtion; ensuring moderate event strengths.
- `truncate_upper:` Truncate the samples of the Inverse Gamma distirubtion; ensuring strong event strengths.
- `print_info:` Switch to turn the infos on/off.
# Examples
```julia
julia> sample_events()
In total, 6 events happend; average event strength 1.06 and a max of 2.3
([0.7524610423212064, 1.1074286946856704, 4.173070960608616, 12.237458469787779, 13.521186230270018, 15.76184312542427], [2.3039812426474797, 0.8953230420965026, 0.87598654113576095, 0.6850390057844807, 0.6750473504611119, 1.6390115288560767])
julia> sample_events(intensity=0.1)
```
"""
function sample_events(myrng::StableRNG; intensity::Real = 0.5, IGammaAlpha::Real = 2, IGammaTheta::Real = 1, tmin::Real = 0.0, tmax::Real= 20.0, truncate_lower::Real = 0.5, truncate_upper::Real = 5.0,  print_info::Bool = false)
    
    numbPoints=rand(myrng, Truncated(Poisson(tmax*intensity), 0, 20)) #Poisson number of points
    event_time=tmax*rand(myrng, numbPoints) .+ tmin#x coordinates of Poisson points
    event_strenght = rand(myrng, Truncated(
                            InverseGamma(IGammaAlpha,IGammaTheta),
                            truncate_lower, truncate_upper), numbPoints)
    # histogram(rand(InverseGamma(1.0,2.0), 1000), xlims = (0,6))
    if print_info
        mean_strength = round(mean(event_strenght),digits=2)
        max_strength = round(maximum(event_strenght),digits=2)
        println("n_events=$numbPoints; x̂=$mean_strength; x_max=$max_strength")
    end
    return sort(event_time), event_strenght # assumes independent event time and strength
end

"""
`update_stressor_level_simulation!()` is the affect function of this model. No condtion is needed since the `PresetTimeCallback` is used, see [here](https://diffeq.sciml.ai/latest/features/callback_functions/#PresetTimeCallback) for more info.

"""
function update_stressor_level_simulation!(integrator)
    if in.(integrator.t, Ref(event_times))
        integrator.u[2] = integrator.u[2] + event_strenghts[findfirst(arg -> arg == integrator.t, event_times)]
    end
end

"""
A simple wrapper to generate a weight matrix with missing completely at random.
"""
function generate_weight_matrix(n::Integer, p_of_missing::AbstractFloat, myrng::StableRNG)

    w_mh = sample(myrng, [0,1], ProbabilityWeights([p_of_missing, (1-p_of_missing)]), n)
    w_sl = sample(myrng, [0,1], ProbabilityWeights([p_of_missing, (1-p_of_missing)]), n)
    
    return BitArray(hcat(w_mh, w_sl)')
end 

"""
`generate_timevarying_data()` is a rather ugly function to simulate data. It mimics the structure of the MARP study to keep simple.
# Arguments
- `n`: the number of elements to compute.
- `rtype`: Resilience type [0,1].
- `ptype`: Parameter change [1,2]. # Could this be 0,1 as well?
- `t_p1`: Range of first baseline period.
- `t_p2`: Range of second baseline period.
- `noise_multiplier`: Float that is sample+(noise_multiplier*randn)
- `percent_item_nonresponse`: percentage of variables with `w=0``
- `event_intensities`: Allows to influence the intensitiy (frequency) of the event sampling. 
- `show_plot`: Shows a pretty plot how the scenarios evolve
# Nomenclature
- `Change types (p?`
    - `p1`: no change
    - `p2`: change in ODE parameter
- `Resileince types (r)`
    - `r0`: resilent
    - `r1`: not resilient
# Debugging
```julia
n = 180
t_p1=0.0:10.0
t_p2=10.0:20.0
myrng = StableRNG(123)
noise_multiplier= 0.1
percent_item_nonresponse=0.1
myseed = 5
curintensity = 0.2
i=1
myrng = Random.seed!(myseed);
rtype = rand(myrng, [0,1], 10);
ptype = rand(myrng, [1,2], 10);
p0  = [-0.3, 0.4, -0.3, -0.1];
p1  = [-0.2, 0.6, -0.2, -0.1];
tsteps = sort(sample(0:0.1:t_p2[end], 10))
prob = ODEProblem(mydiff,[0.1,0.4],(0.0, 18.64),p1);
sol = solve(prob, saveat=tsteps)
# Plots.scatter(solve(remake(prob, p=p0), saveat=tsteps))
```
# Example
```julia
data, r0p1, r0p2, r0p2_worse, r1p1, r1p2, r1p2_worse = simdata = generate_timevarying_data(180, rtype, ptype, 0.0:10.0, 10.0:20.0, 0.1);
data, r0p1, r0p2, r0p2_worse, r1p1, r1p2, r1p2_worse = simdata = generate_timevarying_data(180, rtype, ptype, 0.0:10.0, 10.0:20.0, 0.1);
plot_siumlation(r0p1, r0p2, r0p2_worse, r1p1, r1p2, r1p2_worse)
```
"""
function generate_timevarying_data(odep_r0p0, odep_r0p1, odep_r1p0, odep_r1p1, n::Integer, rtype::AbstractArray, ptype::AbstractArray, t_p1::AbstractArray, t_p2::AbstractArray, myrng::StableRNG, noise_multiplier::AbstractFloat=0.2, percent_item_nonresponse::AbstractFloat=0.1, curintensity::AbstractFloat = 0.25, where_to_save = nothing; kwargs...) 

    all_data = Vector{subject}(undef, n); # init
    
    # putting stuff in the global scope is bad programming behaviour, however, `update_stressor_level!()`, i.e. the affect function, needs it to be that way afaIk
    global t = collect(first(t_p1):1:last(t_p2))
    global W = generate_weight_matrix(size(t,1), percent_item_nonresponse, myrng)
    # global x2 = rand(size(t,1))
    global event_times, event_strenghts = sample_events(myrng)
    
    local local_problem = ODEProblem(mydiff,nothing, # u0
                                (0.0, Float64(last(t_p2))), # tspan
                                nothing, # initial parameters
                                callback = PresetTimeCallback(t, update_stressor_level_simulation!, save_positions=(false,false)));

    # generate data in loop; due to the kiks and the update_stressor_level_simulation function this needs to be redone each time
    for i in 1:n     
        
        obs = replace(1:1:size(t,1), sample(myrng, [9,10,11])=>1) # this mimics the mnpvisno varialbe of MARP, i.e., it counts from 1 to T but signals new baseline obs with 1. Samples 7,8, or 9 and replaces it with 1, just in the actual data this indicates the second baseline obs
        global W = generate_weight_matrix(size(t,1), percent_item_nonresponse, myrng)
        # global kicks = randn(myrng, 21) .+ 0.2 # Shift the kicks slightly upwards to see more action.
        
        # global event_times, event_strenghts = sample_events(kwargs[:intensity]) #kwargs...
        global event_times, event_strenghts = sample_events(myrng, intensity=curintensity) #kwargs...
        global local_problem = ODEProblem(mydiff,nothing, # u0
                                (0.0, Float64(last(t_p2))), # tspan
                                nothing, # initial parameters
                                callback = PresetTimeCallback(event_times, update_stressor_level_simulation!, save_positions=(false,false)));

        # if where_to_save == nothing                                
            where_to_save = t # time_grid for actual data generation; e.g. 0.1 for plotting
        # end
          
        # generate the trajectories in the latent space
        # --------
       global r0p1 = calc_solution(t_p1, [0.0,2.0], odep_r0p0, local_problem, where_to_save; callback = PresetTimeCallback(t, update_stressor_level_simulation!, save_positions=(false,false)))
       global r0p2 = calc_solution(t_p2, r0p1.u[end], odep_r0p0, local_problem, where_to_save; callback = PresetTimeCallback(t, update_stressor_level_simulation!, save_positions=(false,false)))
       
       global r0p1 = calc_solution(t_p1, [0.0,2.0], odep_r0p0, local_problem, where_to_save; callback = PresetTimeCallback(t, update_stressor_level_simulation!, save_positions=(false,false))) 
       global r0p2_worse = calc_solution(t_p2, r0p1.u[end], odep_r0p1, local_problem, where_to_save; callback = PresetTimeCallback(t, update_stressor_level_simulation!, save_positions=(false,false)))

       global r1p1 = calc_solution(t_p1, [0.0,2.0], odep_r1p0, local_problem, where_to_save; callback = PresetTimeCallback(t, update_stressor_level_simulation!, save_positions=(false,false)))
       global r1p2 = calc_solution(t_p2, r1p1.u[end], odep_r1p0, local_problem, where_to_save; callback = PresetTimeCallback(t, update_stressor_level_simulation!, save_positions=(false,false)))

       global r1p1 = moduleODEnet_sp.calc_solution(t_p1, [0.0,2.0], odep_r1p0, local_problem, where_to_save; callback = PresetTimeCallback(t, update_stressor_level_simulation!, save_positions=(false,false)))
       global r1p2_worse = calc_solution(t_p2, r1p1.u[end], odep_r1p1, local_problem, where_to_save; callback = PresetTimeCallback(t, update_stressor_level_simulation!, save_positions=(false,false)))
  
        if rtype[i] == 0 && ptype[i] == 1
            global tmpdata = hcat(Array(r0p1) + randn(myrng, size(Array(r0p1)))*noise_multiplier, Array(r0p2)[:,2:end] + randn(myrng, size(Array(r0p2[:,2:end])))*noise_multiplier) # this Array(r0p2)[:,2:end] is necessary to not "double save" the overlapping point.  
        end
        if rtype[i] == 0 && ptype[i] == 2
            global tmpdata = hcat(Array(r0p1) + randn(myrng, size(Array(r0p1)))*noise_multiplier, Array(r0p2_worse[:,2:end]) + randn(myrng, size(Array(r0p2_worse[:,2:end])))*noise_multiplier)
        end
        if rtype[i] == 1 && ptype[i] == 1
            global tmpdata = hcat(Array(r1p1) + randn(myrng, size(Array(r1p1)))*noise_multiplier, Array(r1p2[:,2:end]) + randn(myrng, size(Array(r1p2[:,2:end])))*noise_multiplier)
        end
        if rtype[i] == 1 && ptype[i] == 2
            global tmpdata = hcat(Array(r1p1) + randn(myrng, size(Array(r1p1)))*noise_multiplier, Array(r1p2_worse[:,2:end]) + randn(myrng, size(Array(r1p2_worse[:,2:end])))*noise_multiplier)
        end
        # --------
        
        # generate the items in the observed space
        # --------
        Xmh = round.(exp.(randn(myrng, size(t,1), 28) .* 0.7 .+ tmpdata[1,:] .* 0.9))
        Xsl = round.(exp.(randn(myrng, size(t,1), 58) .* 0.7 .+ tmpdata[2,:] .* 0.95))
        # Make very sure values stay in the right range
        Xmh = map(x -> ifelse(x > 4.0, 4.0, x), Xmh)
        Xsl = map(x -> ifelse(x > 7, 7.0, x), Xsl)
        # plot(heatmap(Xmh), heatmap(Xsl))
        # --------

        current_subject = conSTRUCTor(i, Xmh, Xsl, t, obs, W)
        current_subject.predVAE = tmpdata

        all_data[i] = current_subject
    end 
    return all_data, r0p1, r0p2, r0p2_worse, r1p1, r1p2, r1p2_worse
end

# mentalh_heatmap_res = heatmap(xmat1, title = "Mental Health (Resilient)", xlabel = "Observation", ylabel = "Items", clim = (1, 4), c = :heat)
# stressors_heatmap_res = heatmap(xmat2, title = "Stressor load (Resilient)", xlabel = "Observation", ylabel = "Items", clim = (1, 7), c = :heat)

# mentalh_heatmap_nonres = heatmap(data[findfirst(x -> x==1, rtype)][1][1], title = "Mental Health (Not resilient)", xlabel = "Observation", ylabel = "Items", clim = (1, 4), c = :heat)
# stressors_heatmap_nonres = heatmap(data[findfirst(x -> x==1, rtype)][1][3], title = "Stressor load (Not resilient)", xlabel = "Observation", ylabel = "Items", clim = (1, 7), c = :heat)

# plot(mentalh_heatmap_res, stressors_heatmap_res, mentalh_heatmap_nonres, stressors_heatmap_nonres, layout = @layout [a b ; c d])


function plot_siumlation(r0p1, r0p2, r0p2_worse, r1p1, r1p2, r1p2_worse)
    
    p=Plots.plot(
            
        Plots.plot(r0p1, ylim=(-1.0,5), title="How it started", legend=false, ylabel = "resilient"),
        Plots.plot(r0p2, ylim=(-1.0,5), title="...would be without\nchange", legend=false),
        Plots.plot(r0p2_worse, ylim=(-1.0,5), title="...would be\nwhen worsened", legend=false), 
        
        Plots.plot(r1p1, ylim=(-1.0,5), legend=false, ylabel = "not resilient"),
        Plots.plot(r1p2, ylim=(-1.0,5), legend=false),
        Plots.plot(r1p2_worse, ylim=(-1.0,5), legend=false),
    layout=(2,3))
    display(p)
end


function extract_parameters(simdata, ODEnet)
    
    results = zeros(2*size(ptype,1), 14); # preallocating result matrix

    for i in 1:size(simdata,1)
                  
        global x1, x2, t, W, obs = simdata[i].predVAE[1,:], simdata[i].predVAE[1,:], simdata[i].data.t, simdata[i].data.W, simdata[i].data.mnpvisno 
        global cur_prob = remake(base_prob, callback = PresetTimeCallback(t, update_stressor_level_simulation!, save_positions=(false,false))) 
        local respondent_ranges = moduleODEnet_sp.split_observation_period_at_first_baseline(obs, W)
        
        ODEp = [ODEnet(moduleODEnet.gather_ODE_info(x1[rg], x2[rg], t[rg], W[:,rg])) for rg in respondent_ranges]
        period_solutions = Array.([calc_solution(t, [0.0,2.0], ODEp[period], cur_prob, [1.0, 2.0, 4.0, 6.0]) for period in 1:size(ODEp,1)])
        for period in 1:size(ODEp,1)
            results[size(ODEp,1)*(i-1) + period, :] = vcat(i, period, rtype[i], ptype[i], ODEp[period], period_solutions[period][1,:])
        end 
    end
    results_df = DataFrame(results, 
    [:id, :p, :rtype, :ptype, :drift_ghq, :dh_on_ghq, :drift_dh, :ghq_on_dh, :intercept_ghq, :intercept_dh, :ptp1, :ptp2, :ptp3, :ptp4])
    
    return results_df
    # comprehensions are extremly slow here; 100.071033 seconds (533.10 M allocations: 25.299 GiB, 7.29% gc time, 99.99% compilation time)
    # [results[i+period-1,:] = vcat(rtype[i], ptype[i], Array.(period_solutions)[period][1,:])  for period in 1:size(ODEp,1)]
    # results = @SMatrix zeros(2*size(ptype,1), 12); # using SMatrix does not immediatly work
 
end

# freqtable(mydf.rtype, mydf.ptype) # missing obs pattern 

function results_long_to_wide(long_df)
    
    long_df.type = vec(hcat(string.(long_df.rtype) .* string.(long_df.ptype)))
    
    pdf_tmp = select(long_df, :id, :p, :drift_ghq, :dh_on_ghq, :drift_dh, :ghq_on_dh)
    
    pdf = hcat( 
        unstack( pdf_tmp, :id, :p, :drift_ghq ,allowduplicates=true),
        select!( unstack( pdf_tmp, :id, :p, :dh_on_ghq  ,allowduplicates=true), Not(:id) ),       
        select!( unstack( pdf_tmp, :id, :p, :drift_dh  ,allowduplicates=true), Not(:id) ),
        select!( unstack( pdf_tmp, :id, :p, :ghq_on_dh  ,allowduplicates=true), Not(:id) ),
        makeunique=true
        )
    
        rename!(pdf, Dict("1.0" => "drift_ghq_p1",    "2.0" => "drift_ghq_p2", 
        "1.0_1" => "dh_on_ghq_p1",  "2.0_1" => "dh_on_ghq_p2", 
        "1.0_2" => "drift_dh_p1",   "2.0_2" => "drift_dh_p2", 
        "1.0_3" => "ghq_on_dh_p1",  "2.0_3" => "ghq_on_dh_p2", 
        )
        )
        return leftjoin(pdf, long_df[!, [:id, :p, :rtype, :ptype, :type]], on = :id)
end

function plot_ode_params(params_df, targets, my_xs, my_dodge; 
    legend_title = "Change type", legend_labels = ["Constant", "Changing"], colors = cgrad(:tab10))

    myxs = params_df[!, my_xs]
    dodge=Int.(params_df[!, my_dodge])
    dodge_values = sort(unique(dodge))

    f = Figure(backgroundcolor = RGBf(0.98, 0.98, 0.98), resolution = (1000, 700))
    
    ax1, bp1 = Makie.boxplot(f[1,1], myxs, params_df[!, targets[1]], dodge = dodge, show_notch = true, 
        color = map(d->d== dodge_values[1] ? colors[1] : colors[2], dodge),
        axis = (xticks = (1:2, ["1", "2"]),
        xlabel = "sub-period",
        title = L"$\alpha$ (drift mh)")
    )
    ax2, bp2 = Makie.boxplot(f[1,2], myxs, params_df[!, targets[2]], dodge = dodge, show_notch = true, 
        color = map(d->d==dodge_values[1] ? colors[1] : colors[2], dodge),
        axis = (xticks = (1:2, ["1", "2"]),
        xlabel = "sub-period",
        title = L"$\beta$ (sl on ml)")
    )
    ax3, bp3 = Makie.boxplot(f[2,1], myxs, params_df[!, targets[3]], dodge = dodge, show_notch = true, 
        color = map(d->d==dodge_values[1] ? colors[1] : colors[2], dodge),
        axis = (xticks = (1:2, ["1", "2"]),
        xlabel = "sub-period",
        title = L"$\delta$ (drift sl)")
    )
    ax4, bp4 = Makie.boxplot(f[2,2], myxs, params_df[!, targets[4]], dodge = dodge, show_notch = true, 
        color = map(d->d==dodge_values[1] ? colors[1] : colors[2], dodge), 
        axis = (xticks = (1:2, ["1", "2"]),
        xlabel = "sub-period",
        title = L"$\eta$ (mh on sl)")
    )
    Makie.Legend(f[1:2,3],  # #Add custom legend
        [PolyElement(polycolor = colors[1]), PolyElement(polycolor = colors[2])], 
        legend_labels, 
        legend_title
    )
    return f
end


function plot_exemplary_subject(ODEnet, simdata, index)

    global z_mh, z_sl, t, W, obs = simdata[index].predVAE[1,:], simdata[index].predVAE[2,:], simdata[index].data.t, simdata[index].data.W, simdata[index].data.mnpvisno;
    ranges = moduleODEnet_sp.split_observation_period_at_first_baseline(obs, W)

    ODEp_sw = [ODEnet(moduleODEnet.gather_ODE_info(z_mh[rg], z_sl[rg], t[rg], W[:,rg])) for rg in ranges]
    sp_solutions = calc_sp_solutions(t, ranges, ODEp_sw, base_prob, 0.1; callback = PresetTimeCallback(t, update_stressor_level!, save_positions=(false,false)))

    plt=Plots.scatter(t[W[1,:].==1], vec(z_mh[W[1,:].==1]), c=[:blue], lab="MH (VAE)")
    Plots.scatter!(plt, t[W[2,:].==1], vec(z_sl[W[2,:].==1]), c=[:red], lab="SL (VAE)")
    Plots.plot!(plt, sp_solutions[1], c=[:blue :red], lab=["GHQ (ODE, sp1)" "DH (ODE, sp1)"])
    Plots.plot!(plt, sp_solutions[2], c=[:darkblue :darkred], lab=["GHQ (ODE, sp2)" "DH (ODE, sp2)"], xlims=(0,20))
    display(plt)
end


# export generate_timevarying_data, plot_siumlation
# end # end of module