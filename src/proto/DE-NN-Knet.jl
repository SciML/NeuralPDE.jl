#using Knet,ArgParse,Main

for p in ("Knet","ArgParse")
    Pkg.installed(p) == nothing && Pkg.add(p)
end

using Knet,ArgParse,Main
#some global functions
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


function train(P, prms, data; epochs =100, iters=10000)
    #print(size(P),size(g),size(prms))
      for epoch=1:epochs
      	      #i = 0
        for (x,y) in data

          #i = i+1
	        g = lossgradient(P, x,y)
          #print(size(P),size(g),size(prms))
          update!(P, g, prms)
          #println(P[1][1],P[2][1],P[3][1],P[4][1])
          if (iters -= 1) <= 0
            return P
          end
        end
      end

    return P
end

function predict(P,x)
  w1, b1, w2, b2 = P
  h = sigm(w1 * x .+ b1)
  return w2 * h .+ b2
end


# function loss(P,x)
#
#     dNx = sum([P[3][i]*P[1][i]*sig_der(P[1][i]*x + P[2][i]) for i = 1:10])
#     ((predict(P,x) + (x*dNx) - f(P,x))[1])^2
# end

function loss(w,x,ygold)
    sumabs2(ygold - predict(w,x)) / size(ygold,2)
end


lossgradient = grad(loss)

function init_params(;ftype=Float32,atype=KnetArray)
    #P = Vector{Vector{Float32}}(4)
    P = Array{Any}(4)
    P[1] = randn(Float32,10,1)
    P[2] = zeros(Float32,10,1)
    P[3] = randn(Float32,1,10)
    P[4] = zeros(Float32,1,1)
    P = map(x -> convert(atype, x), P)

    return P
end

function accuracy(w, dtst, pred=predict)
  ninstance = nerror = 0
  for (x, ygold) in dtst
      ypred = pred(w, x)[1]
      #println(" X: ",x," Prediction: ",ypred[1]," True: ",ygold)
      nerror += abs(ygold - ypred[1])
      ninstance += 1
  end
  return (nerror/ninstance)
end


function params(ws, o)
	prms = Any[]

	for i=1:length(ws)
		w = ws[i]
		if o[:optim] == "Sgd"
			prm = Sgd(;lr=o[:lr])
		elseif o[:optim] == "Momentum"
			prm = Momentum(lr=o[:lr], gamma=o[:gamma])
		elseif o[:optim] == "Adam"
			prm = Adam(lr=o[:lr], beta1=o[:beta1], beta2=o[:beta2], eps=o[:eps])
		elseif o[:optim] == "Adagrad"
			prm = Adagrad(lr=o[:lr], eps=o[:eps])
		elseif o[:optim] == "Adadelta"
			prm = Adadelta(lr=o[:lr], rho=o[:rho], eps=o[:eps])
		elseif o[:optim] == "Rmsprop"
			prm = Rmsprop(lr=o[:lr], rho=o[:rho], eps=o[:eps])
		else
			error("Unknown optimization method!")
		end
		push!(prms, prm)
	end

	return prms
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





function main(args=ARGS)
    s = ArgParseSettings()
    s.description="optimizers.jl (c) Ozan Arkan Can and Deniz Yuret, 2016. Demonstration of different sgd based optimization methods using LeNet."
    s.exc_handler=ArgParse.debug_handler
    @add_arg_table s begin
        ("--seed"; arg_type=Int; default=-1; help="random number seed: use a nonnegative int for repeatable results")
        ("--batchsize"; arg_type=Int; default=1; help="minibatch size")
        ("--lr"; arg_type=Float64; default=0.1; help="learning rate")
	("--eps"; arg_type=Float64; default=1e-6; help="epsilon parameter used in adam, adagrad, adadelta")
	("--gamma"; arg_type=Float64; default=0.95; help="gamma parameter used in momentum")
	("--rho"; arg_type=Float64; default=0.9; help="rho parameter used in adadelta and rmsprop")
	("--beta1"; arg_type=Float64; default=0.9; help="beta1 parameter used in adam")
	("--beta2"; arg_type=Float64; default=0.95; help="beta2 parameter used in adam")
        ("--epochs"; arg_type=Int; default=10; help="number of epochs for training")
        ("--iters"; arg_type=Int; default=6000; help="number of updates for training")
	("--optim"; default="Sgd"; help="optimization method (Sgd, Momentum, Adam, Adagrad, Adadelta, Rmsprop)")
    end
    println(s.description)
    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true)
    println("opts=",[(k,v) for (k,v) in o]...)
    o[:seed] > 0 && srand(o[:seed])
    gpu() >= 0 || error("LeNet only works on GPU machines.")

    #isdefined(MNIST,:xtrn) || MNIST.loaddata()
    global dtrn = batch(x_tr, map(diff_sol,x_tr), o[:batchsize])
    global dtst = batch(x_tst, map(diff_sol,x_tst), o[:batchsize])
    # dtrn = Any[]
    # dtst = Any[]
    # for i=1:10
    #     push!(dtrn, (x_tr[i],map(diff_sol,x_tr)[i]))
    #     push!(dtst, (x_tst[i],map(diff_sol,x_tst)[i]))
    # end
    # #print("params initiated nad trn, tst pushed")
    w = init_params()
    prms = params(w, o)

    # log = Any[]
    # report(epoch)=push!(log, (:epoch,epoch,:trn,accuracy(w,dtrn,predict),:tst,accuracy(w,dtst,predict)))
    report(epoch)=println((:epoch,epoch,:trn,accuracy(w,dtrn,predict),:tst,accuracy(w,dtst,predict)))
    report(0)
    iters = o[:iters]
    #print(size(w),size(g),size(prms))

    @time for epoch=1:o[:epochs]

	    train(w, prms, dtrn; epochs=100, iters=1000)
	    report(epoch)
      # println(w[1][1])
      # println(w[2][1])
      # println(w[3][1])
      # println(w[4][1])
        (iters -= length(dtrn)) <= 0 && break
    end

    # for t in log; println(t); end
    return w
end

main()
