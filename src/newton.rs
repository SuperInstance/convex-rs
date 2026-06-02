//! Newton's method: Hessian-based, damped Newton, convergence analysis.

use crate::DVector;
#[cfg(test)]
use crate::DMatrix;
use crate::convex_functions::Objective;
use serde::{Serialize, Deserialize};

/// Newton step result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewtonResult {
    pub x: DVector<f64>,
    pub value: f64,
    pub iterations: usize,
    pub converged: bool,
    pub gradient_norm: f64,
    pub history: Vec<f64>,
}

/// Newton's method configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewtonMethod {
    pub max_iter: usize,
    pub tol: f64,
    /// Damping parameter α for damped Newton: x_{k+1} = x_k - α * (H^{-1} g)
    pub damping: f64,
    /// Use backtracking line search if true.
    pub line_search: bool,
    /// Backtracking parameters.
    pub bt_alpha: f64,
    pub bt_beta: f64,
    pub bt_c: f64,
}

impl Default for NewtonMethod {
    fn default() -> Self {
        Self {
            max_iter: 100,
            tol: 1e-10,
            damping: 1.0,
            line_search: true,
            bt_alpha: 1.0,
            bt_beta: 0.5,
            bt_c: 1e-4,
        }
    }
}

impl NewtonMethod {
    pub fn new(max_iter: usize, tol: f64) -> Self {
        Self { max_iter, tol, ..Default::default() }
    }

    /// Pure Newton (no line search, damping = 1).
    pub fn pure() -> Self {
        Self {
            line_search: false,
            damping: 1.0,
            ..Default::default()
        }
    }

    /// Damped Newton with given damping factor.
    pub fn damped(damping: f64) -> Self {
        Self {
            damping,
            line_search: false,
            ..Default::default()
        }
    }

    /// Run Newton's method.
    pub fn optimize(&self, f: &dyn Objective, x0: &DVector<f64>) -> NewtonResult {
        let mut x = x0.clone();
        let mut history = Vec::with_capacity(self.max_iter);

        for k in 0..self.max_iter {
            let val = f.eval(&x);
            let grad = f.gradient(&x);
            let grad_norm = grad.norm();
            history.push(val);

            if grad_norm < self.tol {
                return NewtonResult {
                    x,
                    value: val,
                    iterations: k,
                    converged: true,
                    gradient_norm: grad_norm,
                    history,
                };
            }

            let hess = f.hessian(&x);
            // Solve H * d = -g for Newton direction
            let direction = match hess.clone().try_inverse() {
                Some(h_inv) => h_inv * &grad,
                None => {
                    // Fallback: use gradient descent step
                    grad.clone()
                }
            };

            let step_size = if self.line_search {
                self.backtrack(f, &x, &grad, &direction, val)
            } else {
                self.damping
            };

            x = &x - &direction * step_size;
        }

        let val = f.eval(&x);
        let grad = f.gradient(&x);
        history.push(val);

        NewtonResult {
            x,
            value: val,
            iterations: self.max_iter,
            converged: grad.norm() < self.tol,
            gradient_norm: grad.norm(),
            history,
        }
    }

    fn backtrack(
        &self,
        f: &dyn Objective,
        x: &DVector<f64>,
        grad: &DVector<f64>,
        direction: &DVector<f64>,
        f0: f64,
    ) -> f64 {
        let mut t = self.bt_alpha;
        let descent = grad.dot(direction);
        loop {
            let x_new = x - direction * t;
            let f_new = f.eval(&x_new);
            if f_new <= f0 - self.bt_c * t * descent {
                break;
            }
            t *= self.bt_beta;
            if t < 1e-20 {
                break;
            }
        }
        t
    }
}

/// Compute the Newton decrement λ² = g^T H^{-1} g.
/// Used as convergence criterion: λ²/2 < ε implies near-optimality.
pub fn newton_decrement(f: &dyn Objective, x: &DVector<f64>) -> f64 {
    let grad = f.gradient(x);
    let hess = f.hessian(x);
    if let Some(h_inv) = hess.try_inverse() {
        grad.dot(&(h_inv * &grad))
    } else {
        f64::INFINITY
    }
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
    fn test_newton_converges_quadratic() {
        let f = simple_quadratic();
        let nm = NewtonMethod::default();
        let x0 = DVector::from_vec(vec![10.0, 10.0]);
        let result = nm.optimize(&f, &x0);
        assert!(result.converged);
        assert_abs_diff_eq!(result.x[0], 0.5, epsilon = 1e-4);
        assert_abs_diff_eq!(result.x[1], 2.0, epsilon = 1e-4);
    }

    #[test]
    fn test_newton_converges_in_few_iterations() {
        let f = simple_quadratic();
        let nm = NewtonMethod::default();
        let x0 = DVector::from_vec(vec![10.0, 10.0]);
        let result = nm.optimize(&f, &x0);
        // Newton should converge in very few iterations for a quadratic
        assert!(result.iterations <= 10);
    }

    #[test]
    fn test_damped_newton() {
        let f = simple_quadratic();
        let nm = NewtonMethod::damped(0.5);
        let x0 = DVector::from_vec(vec![10.0, 10.0]);
        let result = nm.optimize(&f, &x0);
        assert!(result.converged);
        assert_abs_diff_eq!(result.x[0], 0.5, epsilon = 1e-4);
    }

    #[test]
    fn test_pure_newton() {
        let f = simple_quadratic();
        let nm = NewtonMethod::pure();
        let x0 = DVector::from_vec(vec![10.0, 10.0]);
        let result = nm.optimize(&f, &x0);
        assert!(result.converged);
        // For quadratic, pure Newton should converge in exactly 1 step from any point
        assert!(result.iterations <= 2);
    }

    #[test]
    fn test_newton_decrement() {
        let f = simple_quadratic();
        let x = DVector::from_vec(vec![0.5, 2.0]);
        let dec = newton_decrement(&f, &x);
        assert_abs_diff_eq!(dec, 0.0, epsilon = 1e-4);
    }

    #[test]
    fn test_newton_history_decreasing() {
        let f = simple_quadratic();
        let nm = NewtonMethod::default();
        let x0 = DVector::from_vec(vec![10.0, 10.0]);
        let result = nm.optimize(&f, &x0);
        for w in result.history.windows(2) {
            assert!(w[1] <= w[0] + 1e-6);
        }
    }
}
