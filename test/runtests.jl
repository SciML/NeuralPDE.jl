using SciMLTesting

if current_group() == "NNPDE"
    for group in ("NNPDE1", "NNPDE2")
        withenv("GROUP" => group) do
            run_tests(; test_dir = @__DIR__)
        end
    end
else
    run_tests()
end
