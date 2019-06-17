sigm(x) =  1 ./ (1 .+ exp.(.-x))

function predict(P, x)
    w, b, v = P
    h = sigm(w * x .+ b)
    return v * h
end

function init_params(hl_width)
    w = Flux.param(rand(hl_width,1))
    b = Flux.param(zeros(hl_width,1))
    v = Flux.param(randn(1, hl_width))
    return [w ,b, v]
end

generate_data(low, high, dt) = collect(low:dt:high)
