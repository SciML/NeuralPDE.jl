function gbm_reference_squared_error(u0, alpha, beta, t, prediction, num_samples = 1)
    reference_mean = u0 * exp(alpha * t)
    reference_variance = u0^2 * exp(2alpha * t) * expm1(beta^2 * t)
    return reference_variance / num_samples + abs2(reference_mean - prediction)
end
