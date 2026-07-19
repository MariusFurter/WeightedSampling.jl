# Examples

The repository contains full runnable scripts in `examples/`:

- `examples/linear_regression.jl`
- `examples/eight_schools.jl`
- `examples/2D_ssm.jl`

Run them from the package root:

```bash
julia --project=. examples/linear_regression.jl
julia --project=. examples/eight_schools.jl
julia --project=. examples/2D_ssm.jl
```

Each script defines a model with `@model`, builds an `SMCState`, runs the
transformer with `run!`, and then visualizes posterior summaries.