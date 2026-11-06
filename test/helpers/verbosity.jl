using Logging: Logging, with_logger
using NeuralPDE: NeuralPDE
using Test: Test

const PROGRESS_MARKER = "Verbosity toggle: training_progress"

# Runs `f` and collects NeuralPDE training-progress output from both channels SciMLLogging
# can use: log records (default backend) and stdout (`logging_backend = "core"`).
function capture_progress(f)
    logger = Test.TestLogger(min_level = Logging.Debug)
    result, printed = mktemp() do path, io
        r = redirect_stdout(() -> with_logger(f, logger), io)
        flush(io)
        r, read(path, String)
    end
    records = filter(logger.logs) do r
        r._module === NeuralPDE && startswith(string(r.message), PROGRESS_MARKER)
    end
    return (; result, records, printed)
end

progress_shown(c) = !isempty(c.records) || occursin(PROGRESS_MARKER, c.printed)

# Also rejects the pre-SciMLLogging `@printf`/`println` progress lines.
function progress_hidden(c)
    return isempty(c.records) &&
        !occursin(r"Verbosity toggle: training_progress|Iter: \[|Current loss is", c.printed)
end
