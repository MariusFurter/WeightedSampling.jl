/**
 * Univariate linear-Gaussian discrete-time state-space model.
 *
 * x_0 ~ N(0, x0_std)
 * x_t = a * x_{t-1} + w_t,     w_t ~ N(0, q)
 * y_t ~ N(x_t, r)
 */
model LGSSM1D {
  const a = 0.9
  const q = 1.0
  const r = 0.5
  const x0_std = 1.0

  noise w
  state x
  obs y

  sub initial {
    x ~ gaussian(0.0, x0_std)
  }

  sub transition {
    w ~ gaussian(0.0, q)
    x <- a*x + w
  }

  sub observation {
    y ~ gaussian(x, r)
  }
}