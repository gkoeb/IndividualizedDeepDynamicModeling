using Parameters: @with_kw
@with_kw mutable struct ArgsODEnet
    η = 0.0003              # learning rate
    λ_sp = 2.0              # Penalization multiplier of ODE parameter difference between baseline periods
    λ_ODEp = 0.1           # Penalization multiplier of sum of ODE parameters
    λ_ODEnet = 0.1         # Penalization multiplier of ODEnet parameters
    epochs = 100            # number of epochs
    seed = 110              # random seed
    cuda = false            # use GPU
    input_dim = 14          # number of items
    odep_dim = 4            # number of ODE parameters
    hidden_dim = 5          # hidden dimension
    shuffel_data_per_epoch = true
    verbose_freq = 10       # logging for every verbose_freq iterations
    tblogger = true         # log training with tensorboard
    talk_to_my_repl = true  # switches off all prints to repl, useful when using ProgressMeter and tblogger
    return_data = true
    save_model = false      # save the model parameters to disk
    save_path = "logger/odenet/"    # results path; tensorboard --logdir logger/baselineperiods/
    tbidentifier = "nbruns"    # results path
    finally_plot_all_solutions = false
    prediction_mode = true
end