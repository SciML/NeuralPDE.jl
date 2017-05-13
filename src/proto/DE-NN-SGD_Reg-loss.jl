for p in ("Knet","ArgParse","Compat","GZip")
    Pkg.installed(p) == nothing && Pkg.add(p)
end

using Knet,ArgParse,Compat,GZip


z(P,i,x) = P[1][i]*x + P[2][i]
sig_der(x) = sigm(x)*(1-sigm(x))

#Analytical Solution of the Eq
diff_sol(x) = exp(-(x^2)/2)/(1+x+(x)^3) + x^2

#The phi trial solution
phi(P,x) = 1 + x*(predict(P,x))

#The Differential Equation function for Example 1
f(P,x) = x^3 + 2*x + (x^2)*((1+3*(x^2))/(1+x+(x^3))) - (phi(P,x))*(x + ((1+3*(x^2))/(1+x+x^3)))

 x_tr = linspace(0.0,1,10)
 x_tst = linspace(0.1,1,10)




function predict(P,x)
  w1, b1, w2 = P
  h = sigm(w1 * x .+ b1)
  return w2 * h
end

function loss(w,x,ygold)
    sumabs2(ygold - predict(w,x)) / size(ygold,2)
end

lossgradient = grad(loss)

function train(w, dtrn; lr=.09, epochs=500)
    for epoch=1:epochs
        for (x,y) in dtrn
            g = lossgradient(w, x, y)
            #println(" pred: ",ypred[1]," orig: ",ygold," error: "ypred-ygold)
            for i in 1:length(w)
                # w[i] -= lr * g[i]
                axpy!(-lr, g[i], w[i])
            end
        end
    end
    return w
end

function accuracy(w, dtst, pred=predict)
    ncorrect = ninstance = nerror = 0
    for (x, ygold) in dtst
        ypred = pred(w, x)[1]
        println(" X: ",x," Prediction: ",ypred[1]," True: ",ygold)
        nerror += (ygold - ypred[1])
        ninstance += 1
    end
    return (1-(nerror/ninstance), nerror/ninstance)
end


function init_params(;ftype=Float32,atype=KnetArray)
    #P = Vector{Vector{Float32}}(4)
    P = Array{Any}(3)
    P[1] = randn(Float32,10,1)
    P[2] = zeros(Float32,10,1)
    P[3] = randn(Float32,1,10)
    #P[4] = zeros(Float32,1,1)
    P = map(x -> convert(atype, x), P)

    return P
end

function batch(x, y, batchsize; atype=Array{Float32})
    data = Any[]
    x_ = convert(atype,x)
    y_ = convert(atype,y)
    for i=1:10
        push!(data, (x_[i],y_[i]))
    end
    return data
end




function main(args="")
    s = ArgParseSettings()
    s.description="mnist.jl (c) Deniz Yuret, 2016. Multi-layer perceptron model on the MNIST handwritten digit recognition problem from http://yann.lecun.com/exdb/mnist."
    s.exc_handler=ArgParse.debug_handler
    @add_arg_table s begin
        ("--seed"; arg_type=Int; default=-1; help="random number seed: use a nonnegative int for repeatable results")
        ("--batchsize"; arg_type=Int; default=1; help="minibatch size")
        ("--epochs"; arg_type=Int; default=500; help="number of epochs for training")
        ("--hidden"; nargs='*'; arg_type=Int; help="sizes of hidden layers, e.g. --hidden 128 64 for a net with two hidden layers")
        ("--lr"; arg_type=Float64; default=0.09; help="learning rate")
        ("--winit"; arg_type=Float64; default=0.1; help="w initialized with winit*randn()")
        ("--fast"; action=:store_true; help="skip loss printing for faster run")
        ("--atype"; default=(gpu()>=0 ? "KnetArray{Float32}" : "Array{Float32}"); help="array type: Array for cpu, KnetArray for gpu")
        ("--gcheck"; arg_type=Int; default=0; help="check N random gradients per parameter")
        # These are to experiment with sparse arrays
        # ("--xtype"; help="input array type: defaults to atype")
        # ("--ytype"; help="output array type: defaults to atype")
    end
    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true)
    # if !o[:fast]
    #     println(s.description)
    #     println("opts=",[(k,v) for (k,v) in o]...)
    # end
    o[:seed] > 0 && srand(o[:seed])
    atype = eval(parse(o[:atype]))
    w = init_params()
    #w = weights(o[:hidden]...; atype=atype, winit=o[:winit])
    #if !isdefined(MNIST,:xtrn); loaddata(); end

    global dtrn = batch(x_tr, map(diff_sol,x_tr), o[:batchsize])
    global dtst = batch(x_tst, map(diff_sol,x_tst), o[:batchsize])
    report(epoch)=println((:epoch,epoch,:trn,accuracy(w,dtrn),:tst,accuracy(w,dtst)))
    if o[:fast]
        (train(w, dtrn; lr=o[:lr], epochs=o[:epochs]); gpu()>=0 && Knet.cudaDeviceSynchronize())
    else
        report(0)
        @time for epoch=1:o[:epochs]
            train(w, dtrn; lr=o[:lr], epochs=500)
            report(epoch)
            if o[:gcheck] > 0
                gradcheck(loss, w, first(dtrn)...; gcheck=o[:gcheck], verbose=true)
            end
        end
    end
    return w
end

# This allows both non-interactive (shell command) and interactive calls like:
# $ julia mnist.jl --epochs 10
# julia> MNIST.main("--epochs 10")
if VERSION >= v"0.5.0-dev+7720"
    PROGRAM_FILE == "mnist.jl" && main(ARGS)
else
    !isinteractive() && !isdefined(Core.Main,:load_only) && main(ARGS)
end

main()
