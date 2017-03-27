using Knet

for p in ("Knet","ArgParse")
    Pkg.installed(p) == nothing && Pkg.add(p)
end

using Knet,ArgParse,Main



#Analytical Solution of the Eq
diff_sol(x) = exp(-(x^2)/2)/(1+x+(x)^3) + x^2

#The phi trial solution
phi(P,x) = 1 + x*(predict(P,x))

#The Differential Equation function for Example 1
f(P,x) = x^3 + 2*x + (x^2)*((1+3*(x^2))/(1+x+(x^3))) - (phi(P,x))*(x + ((1+3*(x^2))/(1+x+x^3)))

 x_tr = rand(10)
 x_tst = rand(10)


function train(P, prms, data; epochs=10, iters=6000)
    for epoch=1:epochs
        for (x,y) in data
            g = lossgradient(P, x, y)
            update!(P, g, prms)
            if (iters -= 1) <= 0
                return P
            end
        end
    end
    return P
end

function predict(P,x)
  w1, b1, w2, b2 = P
  h = sigmoid(w1 * x .+ b1)
  return w2 * h .+ b2
end


function loss(P,x,ygold)
    z(i) = P[1][i]*x + P[2][i]
    sig_der(x) = sigmoid(x)(1-sigmoid(x))
    dNx = P[3][i]*P[1][i]*(sig_der(z(i)))
    to_minimize = predict(P,x) + (x*dNx) - f(P,x)
end

lossgradient = grad(loss)

function init_params(;ftype=Float32,atype=KnetArray)
    P = Vector{Vector{Float32}}(4)
    P[1] = randn(Float32,10,1)
    P[2] = zeros(Float32,10,1)
    P[3] = randn(Float32,1,10)
    P[4] = zeros(Float32,1,1)
    return P
end

function main(args)

    P = init_params()
    prm = Adam(lr=0.1, beta1=0.9, beta2=0.95, eps=1e-6)
    prms = Any[]
    push!(prms, prm)
    dtrn = Any[]
    dtst = Any[]
    for i=1:5
        push!(dtrn, (x_trn[i],map(diff_sol,x_trn)[i]))
        push!(dtrn, (x_tst[i],map(diff_sol,x_tst)[i]))
    end

   
    report(epoch)=println((:epoch,epoch,:trn,accuracy(P,dtrn,predict),:tst,accuracy(P,dtst,predict)))
    report(0)
    iters = 1000
    @time for epoch=1:10
	     train(P, prms, dtrn; epochs=1, iters=iters)
	      report(epoch)
        (iters -= length(dtrn)) <= 0 && break
    end

    # for t in log; println(t); end
    return w
end
