using NeuralPDE, Test

struct CallableForDotTest end
(::CallableForDotTest)(x) = x

@testset "Dotting spliced callables" begin
    callable = CallableForDotTest()
    spliced_call = Expr(:call, Expr(:$, callable), :x)

    @test NeuralPDE._dot_(spliced_call) == Expr(:call, callable, :x)
    @test !NeuralPDE.dottable_(Symbol(".+"))

    loop = Expr(:for, Expr(:(=), :x, :xs), Expr(:block, Expr(:call, :sin, :x)))
    dotted_loop = Expr(
        :for, Expr(:(=), :x, :xs), Expr(:block, Expr(:., :sin, Expr(:tuple, :x)))
    )
    @test NeuralPDE._dot_(loop) == dotted_loop
end
