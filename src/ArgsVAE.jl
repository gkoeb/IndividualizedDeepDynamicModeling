using Parameters: @with_kw
@with_kw mutable struct ArgsVAE
    # datatype = "ghq"
    train = true            # should the VAEs be trained? If not, they are only initialized which is necessary for `BSON.@load "model.bson" model`.
    η = 0.003918812               # learning rate
    λ_enc = 0.01             # regularization paramater encoder
    λ_dec = 0.01             # regularization paramater decoder
    batch_size = 32         # batch size
    sample_size = 10        # sampling size for output    
    epochs = 100             # number of epochs
    seed = 1011             # random seed
    cuda = false            # use GPU
    input_dim = 28          # number of items
    latent_dim = 1          # latent dimension (n_z)
    hidden_dim = 10          # hidden dimension
    verbose_freq = 5        # logging for every verbose_freq iterations
    tblogger = false        # log training with tensorboard
    talk_to_my_repl = true # switches off all prints to repl, useful when using ProgressMeter and tblogger
    return_model = true     # return encoder and decoder to global scope
    save_model = false      # save the model parameters to disk
    save_model_object = "myVAE.bson"    # results path
    save_path = "logger/vae/"    # results path
    tbidentifier = "myFirstRuns"    # results path
end