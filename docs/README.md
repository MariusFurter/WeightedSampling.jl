# Documentation Setup for WeightedSampling.jl

This directory contains the documentation setup for WeightedSampling.jl using Documenter.jl.

## Structure

```
docs/
├── Project.toml          # Documentation dependencies
├── make.jl              # Main documentation build script
├── .gitignore           # Ignore build artifacts
└── src/                 # Documentation source files
    ├── index.md         # Homepage
    ├── usage_guide.md   # Detailed usage guide
    ├── api.md           # Complete API reference
    └── examples.md      # Additional examples
```

## Local Development

### Prerequisites

1. Julia 1.6+ installed
2. WeightedSampling.jl package (automatically added as dev dependency)

### Building Documentation

```bash
# Navigate to docs directory
cd docs

# Build documentation
julia --project=. make.jl
```

The built documentation will be available in `docs/build/index.html`.

### Viewing Documentation Locally

Option 1 - Direct file access:
```bash
open build/index.html  # macOS
xdg-open build/index.html  # Linux
```

Option 2 - Local server:
```bash
cd build
python -m http.server 8000
# Then visit http://localhost:8000
```

## Deployment

Documentation is automatically built and deployed to GitHub Pages via GitHub Actions when:
- Commits are pushed to the `main` branch
- Tags are pushed
- Pull requests are opened (build only, no deployment)

The workflow is defined in `.github/workflows/docs.yml`.

### Clean Build

```bash
# Remove build artifacts
rm -rf build/

# Rebuild from scratch
julia --project=. make.jl
```

For more options, see the [Documenter.jl documentation](https://documenter.juliadocs.org/).