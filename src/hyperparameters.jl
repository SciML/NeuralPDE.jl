abstract type AbstractHyperParameter end

# returns an array of tuples of (Array{FastChain}, initial_params)
function getfunction(hyperparam::AbstractHyperParameter, inputdims::AbstractVector{Int}) 
    throw(ArgumentError("getfunction not implemented for this hyperparameter")) 
end

function gettraining(hyperparam::AbstractHyperParameter)
    throw(ArgumentError("gettraining not implemented for this hyperparameter")) 
end

function getseed(hyperparam::AbstractHyperParameter)::Int64
    throw(ArgumentError("getseed not implemented for this hyperparameter")) 
end

function getopt(hyperparam::AbstractHyperParameter)
    throw(ArgumentError("getopt not implemented for this hyperparameter")) 
end

abstract type AbstractNN end

function getfunction(nn::AbstractNN, inputdims::AbstractVector{Int})
    throw(ArgumentError("getfunction not implemented for this nn")) 
end

# this might not be the right design, possibly do a layer over optim & flux optimisers directly, with ParameterSchedulers.jl
abstract type AbstractOptimiser end   

function getopt(optimiser::AbstractOptimiser)
    throw(ArgumentError("getopt not implemented for this optimiser")) 
end

abstract type AbstractHyperParameterSweep end
function generate_hyperparameters(hyperparametersweep::AbstractHyperParameterSweep)
    throw(ArgumentError("generate_hyperparameters not implemented for this hyperparametersweep")) 
end



struct CompositeHyperParameter{NN <: AbstractNN, TRAINING <: NeuralPDE.TrainingStrategies, ADAPTIVELOSS <: AbstractAdaptiveLoss, OPT <: AbstractOptimiser} <: AbstractHyperParameter
    seed::Int64
    nn::NN
    training::TRAINING
    adaptive_loss::ADAPTIVELOSS
    opt::OPT
end

function getfunction(hyperparam::CompositeHyperParameter, inputdims::AbstractVector{Int})
    getfunction(hyperparam.nn, inputdims)
end

function gettraining(hyperparam::CompositeHyperParameter)
    hyperparam.training
end

function getseed(hyperparam::CompositeHyperParameter)
    hyperparam.seed
end

function getadaptiveloss(hyperparam::CompositeHyperParameter)
    hyperparam.adaptive_loss
end

function getopt(hyperparam::CompositeHyperParameter)
    getopt(hyperparam.opt)
end

struct SimpleFeedForwardNetwork{NONLIN, INITIAL_PARAMS} <: AbstractNN
    numhidlayers::Int
    hiddendim::Int
    nonlin::NONLIN
    initial_params::INITIAL_PARAMS
end

function getfunction(feedforwardspec::SimpleFeedForwardNetwork, inputdims::AbstractVector{Int})
    hiddendims = feedforwardspec.hiddendim
    num_hid_layers = feedforwardspec.numhidlayers
    nonlinfunc = getnonlinfunc(feedforwardspec.nonlin)
    initialparamsfunc = getinitialparamsfunc(feedforwardspec.initial_params)
    
    VectorOfMLP(inputdims, hiddendims, num_hid_layers, nonlinfunc, initialparamsfunc)
end


struct SigmoidNonLin end
getnonlinfunc(::SigmoidNonLin) = Flux.σ

struct GELUNonLin end
getnonlinfunc(::GELUNonLin) = Flux.gelu

struct GlorotUniformParams end
getinitialparamsfunc(::GlorotUniformParams) = Flux.glorot_uniform


struct SequenceOfOptimisers{OPTIMISERS} <: AbstractOptimiser
    optimisers::OPTIMISERS
end

function getopt(sequence_of_Optimisers::SequenceOfOptimisers)
    [getopt(optimiser) for optimiser in sequence_of_Optimisers.optimisers]
end

struct BFGSOptimiser <: AbstractOptimiser
    maxiters::Int
end
getopt(bfgs::BFGSOptimiser) = (Optim.BFGS(), bfgs.maxiters)

struct ADAMOptimiser <: AbstractOptimiser
    maxiters::Int
    lr::Float32
end
getopt(adam::ADAMOptimiser) = (Flux.ADAM(adam.lr), adam.maxiters)

struct RADAMOptimiser <: AbstractOptimiser
    maxiters::Int
    lr::Float32
end
getopt(radam::RADAMOptimiser) = (Flux.RADAM(radam.lr), radam.maxiters)






# needs to be able to specify the range of admissable permutations

# structure separate from options?


# needs to be JSON-serializable/deserializable
# when read and processed, generates the same vector of hyperparameters every time


struct StructGenerator{ARGS <: Tuple}
    type::Symbol
    args::ARGS
    function StructGenerator(t, xs...)
        new{typeof(xs)}(t, xs)
    end
end


struct RandomChoice{ARGS <: Union{Tuple, AbstractArray}}
    choices::ARGS
    function RandomChoice(xs...)
        new{typeof(xs)}(xs)
    end
    function RandomChoice(x::AbstractArray)
        new{typeof(x)}(x)
    end
end

function generate_recursion(rng::AbstractRNG, arg)
    generated = 
    if arg isa RandomChoice
        generate_choice(rng, arg)
    elseif arg isa StructGenerator
        generate_struct(rng, arg) 
    elseif arg isa Symbol
        eval(arg)() # treated as a no-arg StructGenerator
    else
        arg # fallback to just providing the arg as-is
    end
    return generated
end

function generate_struct(rng::AbstractRNG, sg::StructGenerator)
    # apply constructor last, recurse into each arg first and generate those
    generated_args = map(arg->generate_recursion(rng, arg), sg.args)
    eval(sg.type)(generated_args...)
end


function generate_choice(rng::AbstractRNG, rc::RandomChoice)
    # randomly pick one a choice, then recurse on its generator/chooser if necessary
    choice = rand(rng, rc.choices)
    generate_recursion(rng, choice)
end

struct StructGeneratorHyperParameterSweep <: AbstractHyperParameterSweep
    hyperseed::Int
    iters::Int
    structgenerator::StructGenerator
end

function generate_hyperparameters(hyperparametersweep::StructGeneratorHyperParameterSweep)
    hyperrng = Random.MersenneTwister(hyperparametersweep.hyperseed)
    hyperparameters = map(_ -> generate_struct(hyperrng, hyperparametersweep.structgenerator), 1:hyperparametersweep.iters)
end
