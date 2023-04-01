using Flux, Tracker
x = [0.8; 0.8]
ann = Chain(Dense(2, 10, tanh), Dense(10, 1))
p, re = Flux.destructure(ann)
z = re(Float64(p))