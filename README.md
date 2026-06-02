# convex-rs

**Convex optimization in Rust.** Every local minimum is a global minimum.

---

Pure Rust implementations of the core algorithms of convex optimization — gradient descent, Newton's method, interior point methods, linear programming, quadratic programming, duality theory, proximal operators, and resource allocation.

~2,900 lines of source, 78 tests, zero unsafe code.

## Install

```toml
[dependencies]
convex-rs = "0.1"
```

## Quick Start

### Minimize a quadratic function

```rust
use convex_rs::{DVector, DMatrix};
use convex_rs::convex_functions::QuadraticFunction;
use convex_rs::gradient_descent::{GradientDescent, StepSize};

let q = DMatrix::from_row_slice(2, 2, &[4.0, 0.0, 0.0, 2.0]);
let c = DVector::from_vec(vec![-2.0, -4.0]);
let f = QuadraticFunction::new(q, c);

let gd = GradientDescent::default();
let result = gd.optimize(&f, &DVector::from_vec(vec![10.0, 10.0]));

println!("x* = {:?}", result.x);       // ≈ [0.5, 2.0]
println!("f(x*) = {}", result.value);   // ≈ -4.5
```

### Solve a linear program

```rust
use convex_rs::linear_programming::{LPProblem, SimplexMethod};

let c = DVector::from_vec(vec![-1.0, -1.0, 0.0]);
let a = DMatrix::from_row_slice(1, 3, &[1.0, 1.0, 1.0]);
let b = DVector::from_vec(vec![1.0]);
let problem = LPProblem::new(c, a, b);

let result = SimplexMethod::default().solve(&problem);
println!("optimal = {}", result.objective_value); // -1.0
```

### Lasso regression

```rust
use convex_rs::proximal::ProximalGradient;

let a = DMatrix::identity(2, 2);
let b = DVector::from_vec(vec![3.0, -2.0]);
let pg = ProximalGradient::accelerated(0.5);
let result = pg.lasso(&a, &b, 1.0, &DVector::zeros(2));
```

## Modules

| Module | Description |
|---|---|
| `convex_functions` | Quadratic objectives, PSD/PD checks, convexity verification, subgradients |
| `convex_sets` | Membership tests and projections for balls, boxes, simplexes, polyhedra |
| `gradient_descent` | GD with fixed / diminishing / backtracking step sizes |
| `newton` | Newton's method (pure, damped, backtracking) |
| `interior_point` | Log-barrier method with central path tracking |
| `linear_programming` | Two-phase simplex method, LP duality |
| `quadratic_programming` | Active-set QP solver, KKT conditions |
| `duality` | Lagrangian mechanics, dual problems, Slater's condition |
| `proximal` | Soft-thresholding, L1/L2 proximal operators, FISTA-accelerated Lasso |
| `resource_allocation` | Resource allocation with utility functions and budget constraints |

## License

MIT OR Apache-2.0
