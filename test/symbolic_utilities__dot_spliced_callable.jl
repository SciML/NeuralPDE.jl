using NeuralPDE, Test
using ModelingToolkit: Differential
using Symbolics: @variables, expand_derivatives

struct CallableForDotTest end
(::CallableForDotTest)(x) = x

@testset "Dotting spliced functions" begin
    spliced_function = :($sin(x))
    callable = CallableForDotTest()
    spliced_callable = :($callable(x))

    @test NeuralPDE._dot_(spliced_function) == Expr(:., sin, Expr(:tuple, :x))
    @test NeuralPDE._dot_(spliced_callable) == Expr(:call, callable, :x)
    @test !NeuralPDE.dottable_(Symbol(".+"))

    loop = Expr(:for, Expr(:(=), :x, :xs), Expr(:block, Expr(:call, :sin, :x)))
    dotted_loop = Expr(
        :for, Expr(:(=), :x, :xs), Expr(:block, Expr(:., :sin, Expr(:tuple, :x)))
    )
    @test NeuralPDE._dot_(loop) == dotted_loop

    @variables x u(..)
    expanded_derivative = expand_derivatives(Differential(x)(u(x)))
    @test !NeuralPDE.is_literal_zero(expanded_derivative)
    @test NeuralPDE.is_literal_zero(0.0)
end
