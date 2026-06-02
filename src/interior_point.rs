//! Interior point methods: barrier functions, central path.

use crate::DVector;
#[cfg(test)]
use crate::DMatrix;
use crate::convex_functions::Objective;
use serde::{Serialize, Deserialize};

/// Type alias for inequality constraint closures.
pub type ConstraintFn = Box<dyn Fn(&DVector<f64>) -> f64 + Send + Sync>;

/// Logarithmic barrier for inequalities g_i(x) <= 0.
/// φ(x) = -Σ log(-g_i(x))
pub struct LogBarrier {
    /// Inequality constraint functions g_i(x) <= 0.
    pub constraints: Vec<ConstraintFn>,
    /// Barrier parameter t (increased along the path).
    pub t: f64,
}

impl LogBarrier {
    pub fn new(
        constraints: Vec<ConstraintFn>,
        t: f64,
    ) -> Self {
        Self { constraints, t }
    }

    /// Evaluate barrier: φ(x) = -Σ log(-g_i(x))
    pub fn eval(&self, x: &DVector<f64>) -> f64 {
        let mut sum = 0.0;
        for g in &self.constraints {
            let gi = g(x);
            if gi >= 0.0 {
                return f64::INFINITY;
            }
            sum += (-gi).ln();
        }
        sum
    }

    /// Gradient of barrier.
    pub fn gradient(&self, x: &DVector<f64>) -> DVector<f64> {
        let n = x.nrows();
        let mut grad = DVector::zeros(n);
        for g in &self.constraints {
            let gi = g(x);
            if gi >= 0.0 {
                return DVector::zeros(n);
            }
            // Numerical gradient of g_i
            let h = 1e-8;
            for j in 0..n {
                let mut xp = x.clone();
                xp[j] += h;
                let mut xm = x.clone();
                xm[j] -= h;
                let gi_grad = (g(&xp) - g(&xm)) / (2.0 * h);
                grad[j] += gi_grad / (-gi);
            }
        }
        grad
    }

    /// Check if x is strictly feasible (all g_i(x) < 0).
    pub fn is_strictly_feasible(&self, x: &DVector<f64>) -> bool {
        self.constraints.iter().all(|g| g(x) < 0.0)
    }
}

/// Result from interior point optimization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IPMResult {
    pub x: DVector<f64>,
    pub value: f64,
    pub iterations: usize,
    pub converged: bool,
    pub central_path: Vec<DVector<f64>>,
    pub barrier_param: f64,
}

/// Interior point method using barrier approach.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteriorPoint {
    pub max_outer: usize,
    pub max_inner: usize,
    pub mu: f64,           // Barrier parameter growth factor
    pub tol: f64,
    pub initial_t: f64,
}

impl Default for InteriorPoint {
    fn default() -> Self {
        Self {
            max_outer: 50,
            max_inner: 100,
            mu: 10.0,
            tol: 1e-8,
            initial_t: 1.0,
        }
    }
}

impl InteriorPoint {
    pub fn new(max_outer: usize, mu: f64, tol: f64) -> Self {
        Self {
            max_outer,
            mu,
            tol,
            ..Default::default()
        }
    }

    /// Minimize f(x) subject to g_i(x) <= 0 using barrier method.
    pub fn optimize(
        &self,
        f: &dyn Objective,
        barrier: &LogBarrier,
        x0: &DVector<f64>,
    ) -> IPMResult {
        let mut x = x0.clone();
        let mut t = self.initial_t;
        let mut central_path = Vec::new();
        let mut converged = false;

        for outer in 0..self.max_outer {
            // Centering step: minimize t*f(x) + φ(x) using Newton's method
            let _barrier_clone = LogBarrier::new(
                // We can't clone closures, so we'll use numerical methods inline
                vec![],
                t,
            );
            // Approximate centering via gradient descent on t*f(x) - barrier
            for _ in 0..self.max_inner {
                let obj_grad = f.gradient(&x);
                let bar_grad = barrier.gradient(&x);
                let combined_grad = &obj_grad * t - &bar_grad;

                if combined_grad.norm() < self.tol {
                    break;
                }

                // Simple gradient step
                let step = 1.0 / (t + 1.0);
                let x_new = &x - &combined_grad * step;

                // Check feasibility
                if barrier.is_strictly_feasible(&x_new) {
                    x = x_new;
                } else {
                    // Reduce step to maintain feasibility
                    let mut s = step;
                    for _ in 0..20 {
                        s *= 0.5;
                        let x_reduced = &x - &combined_grad * s;
                        if barrier.is_strictly_feasible(&x_reduced) {
                            x = x_reduced;
                            break;
                        }
                    }
                }
            }

            central_path.push(x.clone());

            // Check duality gap: m/t < tolerance (m = number of constraints)
            let m = barrier.constraints.len();
            let gap = m as f64 / t;
            if gap < self.tol {
                converged = true;
                let val = f.eval(&x);
                return IPMResult {
                    x,
                    value: val,
                    iterations: outer + 1,
                    converged,
                    central_path,
                    barrier_param: t,
                };
            }

            t *= self.mu;
        }

        let val = f.eval(&x);
        IPMResult {
            x,
            value: val,
            iterations: self.max_outer,
            converged,
            central_path,
            barrier_param: t,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::convex_functions::QuadraticFunction;
    use approx::assert_abs_diff_eq;

    fn _box_constraints(n: usize) -> Vec<Box<dyn Fn(&DVector<f64>) -> f64 + Send + Sync>> {
        let mut constraints = Vec::new();
        for i in 0..n {
            constraints.push(Box::new(move |x: &DVector<f64>| x[i] - 1.0) as Box<dyn Fn(&DVector<f64>) -> f64 + Send + Sync>);
            constraints.push(Box::new(move |x: &DVector<f64>| -x[i] - 0.5) as Box<dyn Fn(&DVector<f64>) -> f64 + Send + Sync>);
        }
        constraints
    }

    #[test]
    fn test_barrier_eval() {
        let constraints: Vec<Box<dyn Fn(&DVector<f64>) -> f64 + Send + Sync>> = vec![
            Box::new(|x: &DVector<f64>| x[0] - 1.0),
            Box::new(|x: &DVector<f64>| -x[0] - 0.5),
        ];
        let barrier = LogBarrier::new(constraints, 1.0);
        let x = DVector::from_vec(vec![0.0]);
        let val = barrier.eval(&x);
        assert!(val.is_finite());
    }

    #[test]
    fn test_barrier_infeasible() {
        let constraints: Vec<Box<dyn Fn(&DVector<f64>) -> f64 + Send + Sync>> = vec![
            Box::new(|x: &DVector<f64>| x[0] - 1.0),
        ];
        let barrier = LogBarrier::new(constraints, 1.0);
        let x = DVector::from_vec(vec![2.0]);
        assert!(!barrier.is_strictly_feasible(&x));
    }

    #[test]
    fn test_barrier_feasible() {
        let constraints: Vec<Box<dyn Fn(&DVector<f64>) -> f64 + Send + Sync>> = vec![
            Box::new(|x: &DVector<f64>| x[0] - 1.0),
            Box::new(|x: &DVector<f64>| -x[0] - 1.0),
        ];
        let barrier = LogBarrier::new(constraints, 1.0);
        let x = DVector::from_vec(vec![0.0]);
        assert!(barrier.is_strictly_feasible(&x));
    }

    #[test]
    fn test_barrier_gradient() {
        let constraints: Vec<Box<dyn Fn(&DVector<f64>) -> f64 + Send + Sync>> = vec![
            Box::new(|x: &DVector<f64>| x[0] - 1.0),
            Box::new(|x: &DVector<f64>| -x[0] - 1.0),
        ];
        let barrier = LogBarrier::new(constraints, 1.0);
        let x = DVector::from_vec(vec![0.5]);
        let grad = barrier.gradient(&x);
        assert!(grad.norm() > 0.0);
    }

    #[test]
    fn test_ipm_simple() {
        let q = DMatrix::from_row_slice(1, 1, &[2.0]);
        let c = DVector::from_vec(vec![-1.0]);
        let f = QuadraticFunction::new(q, c);

        let constraints: Vec<Box<dyn Fn(&DVector<f64>) -> f64 + Send + Sync>> = vec![
            Box::new(|x: &DVector<f64>| x[0] - 2.0),   // x <= 2
            Box::new(|x: &DVector<f64>| -x[0] - 1.0),   // x >= -1
        ];
        let barrier = LogBarrier::new(constraints, 1.0);

        let ipm = InteriorPoint::new(50, 10.0, 1e-6);
        let x0 = DVector::from_vec(vec![0.0]);
        let result = ipm.optimize(&f, &barrier, &x0);
        // Unconstrained optimum is x = 0.5, which is feasible
        assert_abs_diff_eq!(result.x[0], 0.5, epsilon = 0.1);
    }

    #[test]
    fn test_ipm_central_path() {
        let q = DMatrix::from_row_slice(1, 1, &[2.0]);
        let c = DVector::from_vec(vec![-3.0]);
        let f = QuadraticFunction::new(q, c);

        let constraints: Vec<Box<dyn Fn(&DVector<f64>) -> f64 + Send + Sync>> = vec![
            Box::new(|x: &DVector<f64>| x[0] - 1.0),   // x <= 1
            Box::new(|x: &DVector<f64>| -x[0]),          // x >= 0
        ];
        let barrier = LogBarrier::new(constraints, 1.0);

        let ipm = InteriorPoint::new(30, 5.0, 1e-6);
        let x0 = DVector::from_vec(vec![0.5]);
        let result = ipm.optimize(&f, &barrier, &x0);
        // Unconstrained optimum is x = 1.5, but constraint x <= 1 means x* = 1
        assert!(result.central_path.len() > 0);
    }
}
