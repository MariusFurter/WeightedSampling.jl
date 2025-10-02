WeightedSampling provides the `@smc` macro that allows for quick and flexible specification of sequential Monte Carlo sampling schemes.

Section describing SMCKernel type
- `sampler : (in_args) -> sample` generates random samples
- `weighter : (in_args, sample) -> log_weight` weighs a given sample
- `logpdf : (in_args, sample) -> log_pdf` expresses the log_pdf of the weighted kernel. If sampler $\int_x weighter(x | in_args) f(x | in_args)$ where f is the density of `sampler`.

`@smc` allows you to write programs based on SMCKernels
- Assign `.=`
- Sample `~`
- Observe `=>`
- Move `<<`
- For loops
- If statements

Other Julia statements like assignment `=` will be left as is and define local variables.

Any data you want to access or modify should be passed as arguments to the smc function.

Vector component `[]` should work as expected, even on particle variables.

Index interpolation with `{}`. 

Section describing reserved names and how to use them.
`weights` is not an allowed variable name
Local variables in body `particles`, `resampled`, `ess_perc`, `evidence`.

Section on performance.

Multivariate dists can be sped up by (directly!) outputting static arrays of fixed size. E.g 2DNormal, etc.