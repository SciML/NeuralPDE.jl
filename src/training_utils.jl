

function predict(P,x)
  w1, b1, w2 = P
  h = sigm(w1 * x .+ b1)
  return w2 * h
end

function loss_reg(w,x,ygold)
    sumabs2(ygold - predict(w,x)) / size(ygold,2)
end

function loss_trial(P,x,ygold)
  dNx = sum([P[3][i]*P[1][i]*sig_der(P[1][i]*x .+ P[2][i]) for i = 1:10])
  ((predict(P,x)+(x*dNx)-f(P,x))[1])^2
end



function train(P, prms, data, regFlag=true; epochs =100, iters=10000)
    #print(size(P),size(g),size(prms))
    if regFlag
      lossgradient = grad(loss_reg)
    else
      lossgradient = grad(loss_trial)
    end
      for epoch=1:epochs
        for (x,y) in data
	        g = lossgradient(P,x,y)
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

function accuracy(w, dtst, pred=predict)
    ncorrect = ninstance = nerror = 0
    for (x, ygold) in dtst
        ypred = pred(w, x)[1]
        #println(" X: ",x," Prediction: ",ypred[1]," True: ",ygold)
        nerror += abs(ygold - ypred[1])
        ninstance += 1
    end
    return (nerror/ninstance)
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




function generate_data(low, high, dt; atype=Array{Float32})
    data = Any[]
    num_points = 1/dt
    x = linspace(low,high,num_points)
    x_ = convert(atype,x)
    # for i=1:num_points
    #     push!(data, (x_[i],y_[i]))
    # end
    return x_
end
