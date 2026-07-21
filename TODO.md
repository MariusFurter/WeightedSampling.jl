# WeightedSampling.jl - TODOs

- [ ] Weight transformers incorreclty implemented.
- [ ] autoRW does not accept kwargs besides `diversity`.

## Convenience

## Composition and recursion of SMC samplers
- [ ] **Recursion & composition**: Enable composition with other SMC functions more generally
- [ ] **Type wrapping**: Wrap SMC function in type and implement case distinction in code
- [ ] **Vector arguments**: Verify that @model generated function works with vector arguments
- [ ] **Return value specification**: Add a way to specify custom return values
- [ ] **Evidence composition**: Determine how evidence should compose across function calls
- [ ] **Move handling**: Define how moves should work in composed functions

## Documentation

## Examples
- [ ] Stochastic volatility
- [ ] Parameter inference in SSM

## Testing & Validation
- [ ] **Expand test coverage**: Add more comprehensive tests
- [ ] **Benchmark suite**: Compare performance against other SMC implementations
- [ ] **Type stability**: Add type stability checks

## Algorithm Extensions
- [ ] **Multithreading support** 
- [ ] **Additional resampling schemes**: Support more resampling algorithms
- [ ] **Pseudo-marginal sampling**: Add support for pseudo-marginal sampling when some variables are overwritten
- [ ] **Adaptive algorithms**: Implement adaptive SMC variants