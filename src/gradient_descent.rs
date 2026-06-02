//! Gradient descent with step size selection and convergence analysis.

use crate::{DVector, DMatrix};
use crate::convex_functions::Objective;
use serde::{Serialize, Deserialize};

/// Step size strategy for gradient descent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StepSize {
    /// Fixed step size.
    Fixed(f64),
    /// Diminishing: α_k = α₀ / (1 + β*k)
    Diminishing { alpha0: f64, beta: f64 },
    /// Backtracking line search (Armijo).
    Backtracking { alpha: f64, beta: f64, c: f64 },
}

impl Default for StepSize {
    fn default() -> Self {
        StepSize::Backtracking {
            alpha: 1.0,
            beta: 0.8,
            c: 1e-4,
        }
    }
}

/// Result of gradient descent optimization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GDResult {
    pub x: DVector<f64>,
    pub value: f64,
    pub iterations: usize,
    pub converged: bool,
    pub gradient_norm: f64,
    pub history: Vec<f64>,
}

/// Gradient descent optimizer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientDescent {
    pub step_size: StepSize,
    pub max_iter: usize,
    pub tol: f64,
}

impl Default for GradientDescent {
    fn default() -> Self {
        Self {
            step_size: StepSize::default(),
            max_iter: 10000,
            tol: 1e-8,
        }
    }
}

impl GradientDescent {
    pub fn new(step_size: StepSize, max_iter: usize, tol: f64) -> Self {
        Self { step_size, max_iter, tol }
    }

    /// Run gradient descent.
    pub fn optimize(&self, f: &dyn Objective, x0: &DVector<f64>) -> GDResult {
        let mut x = x0.clone();
        let mut history = Vec::with_capacity(self.max_iter);
        let mut converged = false;

        for k in 0..self.max_iter {
            let val = f.eval(&x);
            let grad = f.gradient(&x);
            let grad_norm = grad.norm();

            history.push(val);

            if grad_norm < self.tol {
                converged = true;
                return GDResult {
                    x,
                    value: val,
                    iterations: k,
                    converged,
                    gradient_norm: grad_norm,
                    history,
                };
            }

            let alpha = self.compute_step(f, &x, &grad, k);

            x = &x - &grad * alpha;
        }

        let val = f.eval(&x);
        let grad = f.gradient(&x);
        history.push(val);

        GDResult {
            x,
            value: val,
            iterations: self.max_iter,
            converged,
            gradient_norm: grad.norm(),
            history,
        }
    }

    fn compute_step(&self, f: &dyn Objective, x: &DVector<f64>, grad: &DVector<f64>, k: usize) -> f64 {
        match &self.step_size {
            StepSize::Fixed(a) => *a,
            StepSize::Diminishing { alpha0, beta } => alpha0 / (1.0 + beta * k as f64),
            StepSize::Backtracking { alpha, beta, c } => {
                let mut t = *alpha;
                let f0 = f.eval(x);
                let descent = grad.norm_squared();
                loop {
                    let x_new = x - grad * t;
                    let f_new = f.eval(&x_new);
                    if f_new <= f0 - c * t * descent {
                        break;
                    }
                    t *= beta;
                    if t < 1e-20 {
                        break;
                    }
                }
                t
            }
        }
    }
}

/// Estimate the convergence rate from iteration history.
/// For convex functions, returns the empirical rate (O(1/k) for convex, O(ρ^k) for strongly convex).
pub fn estimate_convergence_rate(history: &[f64]) -> f64 {
    if history.len() < 10 {
        return f64::NAN;
    }
    let n = history.len();
    let first_half = &history[..n / 2];
    let second_half = &history[n / 2..];
    let avg_first: f64 = first_half.iter().sum::<f64>() / first_half.len() as f64;
    let avg_second: f64 = second_half.iter().sum::<f64>() / second_half.len() as f64;
    if avg_first.abs() < 1e-15 {
        return 0.0;
    }
    (avg_second - avg_first) / avg_first
}

/// Compute the Lipschitz constant of the gradient from a quadratic function's Hessian.
pub fn lipschitz_constant(hessian: &DMatrix<f64>) -> f64 {
    let sym = (hessian + hessian.transpose()) * 0.5;
    let eig = sym.symmetric_eigen();
    eig.eigenvalues.iter().copied().fold(f64::NEG_INFINITY, f64::max).max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::convex_functions::QuadraticFunction;
    use approx::assert_abs_diff_eq;

    fn simple_quadratic() -> QuadraticFunction {
        let q = DMatrix::from_row_slice(2, 2, &[4.0, 0.0, 0.0, 2.0]);
        let c = DVector::from_vec(vec![-2.0, -4.0]);
        QuadraticFunction::new(q, c)
    }

    #[test]
    fn test_gd_fixed_step_converges() {
        let f = simple_quadratic();
        let gd = GradientDescent::new(StepSize::Fixed(0.05), 10000, 1e-10);
        let x0 = DVector::from_vec(vec![10.0, 10.0]);
        let result = gd.optimize(&f, &x0);
        assert!(result.converged);
        assert_abs_diff_eq!(result.x[0], 0.5, epsilon = 1e-3);
        assert_abs_diff_eq!(result.x[1], 2.0, epsilon = 1e-3);
    }

    #[test]
    fn test_gd_backtracking_converges() {
        let f = simple_quadratic();
        let gd = GradientDescent::new(
            StepSize::Backtracking { alpha: 1.0, beta: 0.5, c: 1e-4 },
            10000,
            1e-10,
        );
        let x0 = DVector::from_vec(vec![10.0, 10.0]);
        let result = gd.optimize(&f, &x0);
        assert!(result.converged);
        assert_abs_diff_eq!(result.x[0], 0.5, epsilon = 1e-3);
        assert_abs_diff_eq!(result.x[1], 2.0, epsilon = 1e-3);
    }

    #[test]
    fn test_gd_diminishing_step() {
        let f = simple_quadratic();
        let gd = GradientDescent::new(
            StepSize::Diminishing { alpha0: 0.5, beta: 0.01 },
            10000,
            1e-8,
        );
        let x0 = DVector::from_vec(vec![5.0, 5.0]);
        let result = gd.optimize(&f, &x0);
        assert!(result.converged);
        assert_abs_diff_eq!(result.x[0], 0.5, epsilon = 1e-3);
    }

    #[test]
    fn test_gd_history_decreasing() {
        let f = simple_quadratic();
        let gd = GradientDescent::new(
            StepSize::Backtracking { alpha: 1.0, beta: 0.8, c: 1e-4 },
            1000,
            1e-12,
        );
        let x0 = DVector::from_vec(vec![10.0, 10.0]);
        let result = gd.optimize(&f, &x0);
        // Values should be (approximately) non-increasing
        for w in result.history.windows(2) {
            assert!(w[1] <= w[0] + 1e-6, "History not decreasing: {} > {}", w[1], w[0]);
        }
    }

    #[test]
    fn test_lipschitz_constant() {
        let q = DMatrix::from_row_slice(2, 2, &[4.0, 0.0, 0.0, 2.0]);
        let l = lipschitz_constant(&q);
        assert_abs_diff_eq!(l, 4.0, epsilon = 1e-10);
    }

    #[test]
    fn test_convergence_rate_estimated() {
        let f = simple_quadratic();
        let gd = GradientDescent::new(StepSize::Fixed(0.1), 5000, 1e-10);
        let x0 = DVector::from_vec(vec![10.0, 10.0]);
        let result = gd.optimize(&f, &x0);
        // For strongly convex, should converge
        let rate = estimate_convergence_rate(&result.history);
        assert!(rate.is_finite());
    }
}
