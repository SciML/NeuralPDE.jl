function get_likelihood_estimate_function(discretization::BayesianPINN)
	dataset_pde, dataset_bc = discretization.dataset

	pde_loss_functions, bc_loss_functions = merge_strategy_with_loglikelihood_function(
		pinnrep, strategy,
		datafree_pde_loss_functions, datafree_bc_loss_functions)

	# required as Physics loss also needed on the discrete dataset domain points
	# data points are discrete and so by default GridTraining loss applies
	# passing placeholder dx with GridTraining, it uses data points irl
	datapde_loss_functions, databc_loss_functions = if dataset_bc !== nothing ||
													   dataset_pde !== nothing
		merge_strategy_with_loglikelihood_function(pinnrep, GridTraining(0.1),
			datafree_pde_loss_functions, datafree_bc_loss_functions,
			train_sets_pde = dataset_pde, train_sets_bc = dataset_bc)
	else
		nothing, nothing
	end

	# this includes losses from dataset domain points as well as discretization points
	function full_loss_function(θ, allstd::Vector{Vector{Float64}})
		stdpdes, stdbcs, stdextra = allstd
		# the aggregation happens on cpu even if the losses are gpu, probably fine since it's only a few of them
		# SSE FOR LOSS ON GRIDPOINTS not MSE ! i, j depend on number of bcs and eqs
		pde_loglikelihoods = sum([pde_loglike_function(θ, stdpdes[i])
								  for (i, pde_loglike_function) in enumerate(pde_loss_functions)])

		bc_loglikelihoods = sum([bc_loglike_function(θ, stdbcs[j])
								 for (j, bc_loglike_function) in enumerate(bc_loss_functions)])

		# final newloss creation components are similar to this
		if !(datapde_loss_functions isa Nothing)
			pde_loglikelihoods += sum([pde_loglike_function(θ, stdpdes[j])
									   for (j, pde_loglike_function) in enumerate(datapde_loss_functions)])
		end

		if !(databc_loss_functions isa Nothing)
			bc_loglikelihoods += sum([bc_loglike_function(θ, stdbcs[j])
									  for (j, bc_loglike_function) in enumerate(databc_loss_functions)])
		end

		# this is kind of a hack, and means that whenever the outer function is evaluated the increment goes up, even if it's not being optimized
		# that's why we prefer the user to maintain the increment in the outer loop callback during optimization
		@ignore_derivatives if self_increment
			iteration[] += 1
		end

		@ignore_derivatives begin
			reweight_losses_func(θ, pde_loglikelihoods,
				bc_loglikelihoods)
		end

		weighted_pde_loglikelihood = adaloss.pde_loss_weights .* pde_loglikelihoods
		weighted_bc_loglikelihood = adaloss.bc_loss_weights .* bc_loglikelihoods

		sum_weighted_pde_loglikelihood = sum(weighted_pde_loglikelihood)
		sum_weighted_bc_loglikelihood = sum(weighted_bc_loglikelihood)
		weighted_loglikelihood_before_additional = sum_weighted_pde_loglikelihood +
												   sum_weighted_bc_loglikelihood

		full_weighted_loglikelihood = if additional_loss isa Nothing
			weighted_loglikelihood_before_additional
		else
			(θ_, p_) = param_estim ? (θ.depvar, θ.p) : (θ, nothing)
			_additional_loss = additional_loss(phi, θ_, p_)
			_additional_loglikelihood = logpdf(Normal(0, stdextra), _additional_loss)

			weighted_additional_loglikelihood = adaloss.additional_loss_weights[1] *
												_additional_loglikelihood

			weighted_loglikelihood_before_additional + weighted_additional_loglikelihood
		end

		return full_weighted_loglikelihood
	end

	return full_loss_function
end
