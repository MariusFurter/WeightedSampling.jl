using Documenter
using WeightedSampling

# Set up the documentation build
makedocs(;
    modules=[WeightedSampling],
    sitename="WeightedSampling.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://MariusFurter.github.io/WeightedSampling.jl",
        edit_link="main",
        assets=String[],
        repolink="https://github.com/MariusFurter/WeightedSampling.jl",
    ),
    pages=[
        "Home" => "index.md",
        "Usage Guide" => "usage_guide.md",
        "API Reference" => "api.md",
        "Examples" => "examples.md",
    ],
    repo="https://github.com/MariusFurter/WeightedSampling.jl/blob/{commit}{path}#{line}",
    checkdocs=:exports,  # Require all exported functions to have docstrings
)

# Deploy documentation to gh-pages branch
deploydocs(;
    repo="github.com/MariusFurter/WeightedSampling.jl",
    devbranch="main",
)