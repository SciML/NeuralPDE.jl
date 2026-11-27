using Logging: Logging, with_logger
using NeuralPDE: NeuralPDE
using SciMLLogging: get_logging_backend
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

function progress_counters(c, solver, label = "Iter"; backend = get_logging_backend())
    messages = if backend == "core"
        (c.printed,)
    else
        all(r -> r.level == Logging.Info, c.records) || return Int[]
        (string(r.message) for r in c.records)
    end
    pattern = Regex(
        PROGRESS_MARKER * raw"[^\S\n]*\n[^\S\n]*\[" * solver * raw"\]\s+" * label *
            raw": \[ *(\d+)(?:/\d+)?\]\s+Loss: (\S+)$", "m"
    )
    counters = Int[]
    for message in messages, m in eachmatch(pattern, message)
        loss = tryparse(Float64, m[2])
        if loss !== nothing && isfinite(loss)
            push!(counters, parse(Int, m[1]))
        end
    end
    return counters
end

function progress_shown(c, solver; backend = get_logging_backend())
    return !isempty(progress_counters(c, solver; backend))
end

function progress_hidden(c)
    return isempty(c.records) &&
        !occursin(r"Verbosity toggle: training_progress|Iter: \[|Callback: \[", c.printed)
end
