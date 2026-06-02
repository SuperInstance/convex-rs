//! Proximal operators: soft thresholding, proximal gradient method.

use crate::{DVector, DMatrix};
use crate::convex_functions::Objective;
use serde::{Serialize, Deserialize};

/// Soft thresholding operator: prox_{λ||·||_1}(x) = sign(x) * max(|x| - λ, 0).
pub fn soft_threshold(x: &DVector<f64>, lambda: f64) -> DVector<f64> {
    let data: Vec<f64> = x.iter()
        .map(|&xi| {
            if xi > lambda {
                xi - lambda
            } else if xi < -lambda {
                xi + lambda
            } else {
                0.0
            }
        })
        .collect();
    DVector::from_vec(data)
}

/// Proximal operator for L2 norm: prox_{λ||·||_2}(x) = x * max(1 - λ/||x||, 0).
pub fn prox_l2(x: &DVector<f64>, lambda: f64) -> DVector<f64> {
    let norm = x.norm();
    if norm <= lambda {
        DVector::zeros(x.nrows())
    } else {
        x * (1.0 - lambda / norm)
    }
}

/// Proximal operator for quadratic: prox_{f}(x) where f(y) = 0.5 (y-z)^T Q (y-z).
/// = (I + tQ)^{-1} (x + t Q z)
pub fn prox_quadratic(x: &DVector<f64>, q: &DMatrix<f64>, t: f64) -> DVector<f64> {
    let n = x.nrows();
    let i_plus_tq = DMatrix::identity(n, n) + q * t;
    match i_plus_tq.try_inverse() {
        Some(inv) => inv * x,
        None => x.clone(),
    }
}

/// Result of proximal gradient optimization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProxGradResult {
    pub x: DVector<f64>,
    pub objective_value: f64,
    pub iterations: usize,
    pub converged: bool,
    pub gradient_norm: f64,
    pub history: Vec<f64>,
}

/// Proximal gradient method for min f(x) + g(x), where f is smooth and g has a proximal operator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProximalGradient {
    pub max_iter: usize,
    pub tol: f64,
    pub step_size: f64,
    pub use_acceleration: bool,
}

impl Default for ProximalGradient {
    fn default() -> Self {
        Self {
            max_iter: 5000,
            tol: 1e-8,
            step_size: 0.01,
            use_acceleration: false,
        }
    }
}

impl ProximalGradient {
    pub fn new(max_iter: usize, tol: f64, step_size: f64) -> Self {
        Self {
            max_iter,
            tol,
            step_size,
            use_acceleration: false,
        }
    }

    /// Use FISTA acceleration (Beck & Teboulle).
    pub fn accelerated(step_size: f64) -> Self {
        Self {
            step_size,
            use_acceleration: true,
            ..Default::default()
        }
    }

    /// Lasso: min (1/2n)||Ax - b||^2 + λ||x||_1
    pub fn lasso(
        &self,
        a: &DMatrix<f64>,
        b: &DVector<f64>,
        lambda: f64,
        x0: &DVector<f64>,
    ) -> ProxGradResult {
        let n = a.nrows() as f64;
        let ata = a.transpose() * a / n;
        let atb = a.transpose() * b / n;

        let mut x = x0.clone();
        let mut y = x.clone();
        let mut t = 1.0;
        let mut history = Vec::with_capacity(self.max_iter);

        for k in 0..self.max_iter {
            let obj_val = {
                let residual = a * &y - b;
                0.5 * residual.norm_squared() / n + lambda * y.iter().map(|v| v.abs()).sum::<f64>()
            };
            history.push(obj_val);

            let grad = &ata * &y - &atb;
            let grad_norm = grad.norm();

            if grad_norm < self.tol && k > 0 {
                let residual = a * &x - b;
                let final_obj = 0.5 * residual.norm_squared() / n + lambda * x.iter().map(|v| v.abs()).sum::<f64>();
                return ProxGradResult {
                    x,
                    objective_value: final_obj,
                    iterations: k,
                    converged: true,
                    gradient_norm: grad_norm,
                    history,
                };
            }

            // Gradient step + proximal (soft thresholding)
            let z = &y - &grad * self.step_size;
            let x_new = soft_threshold(&z, lambda * self.step_size);

            if self.use_acceleration {
                let t_new = (1.0_f64 + (1.0_f64 + 4.0_f64 * t * t).sqrt()) / 2.0_f64;
                let beta = (t - 1.0) / t_new;
                y = &x_new + (&x_new - &x) * beta;
                t = t_new;
                x = x_new;
            } else {
                x = x_new;
                y = x.clone();
            }
        }

        let residual = a * &x - b;
        let final_obj = 0.5 * residual.norm_squared() / n + lambda * x.iter().map(|v| v.abs()).sum::<f64>();
        let grad = &ata * &x - &atb;

        ProxGradResult {
            x,
            objective_value: final_obj,
            iterations: self.max_iter,
            converged: grad.norm() < self.tol,
            gradient_norm: grad.norm(),
            history,
        }
    }

    /// Generic proximal gradient with custom smooth objective and proximal operator.
    pub fn optimize(
        &self,
        f: &dyn Objective,
        prox: &dyn Fn(&DVector<f64>) -> DVector<f64>,
        x0: &DVector<f64>,
    ) -> ProxGradResult {
        let mut x = x0.clone();
        let mut history = Vec::with_capacity(self.max_iter);

        for k in 0..self.max_iter {
            let val = f.eval(&x);
            history.push(val);

            let grad = f.gradient(&x);
            let grad_norm = grad.norm();

            if grad_norm < self.tol {
                return ProxGradResult {
                    x,
                    objective_value: val,
                    iterations: k,
                    converged: true,
                    gradient_norm: grad_norm,
                    history,
                };
            }

            let z = &x - &grad * self.step_size;
            x = prox(&z);
        }

        let val = f.eval(&x);
        ProxGradResult {
            x: x.clone(),
            objective_value: val,
            iterations: self.max_iter,
            converged: false,
            gradient_norm: f.gradient(&x).norm(),
            history,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_soft_threshold_positive() {
        let x = DVector::from_vec(vec![3.0, -2.0, 0.5]);
        let result = soft_threshold(&x, 1.0);
        assert_abs_diff_eq!(result[0], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[1], -1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[2], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_soft_threshold_large_lambda() {
        let x = DVector::from_vec(vec![0.3, -0.2, 0.1]);
        let result = soft_threshold(&x, 1.0);
        assert_abs_diff_eq!(result[0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[1], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[2], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_prox_l2_large_lambda() {
        let x = DVector::from_vec(vec![1.0, 1.0]);
        let result = prox_l2(&x, 10.0);
        assert_abs_diff_eq!(result[0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[1], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_prox_l2_small_lambda() {
        let x = DVector::from_vec(vec![3.0, 4.0]);
        let result = prox_l2(&x, 1.0);
        // ||x|| = 5, factor = 1 - 1/5 = 0.8
        assert_abs_diff_eq!(result[0], 2.4, epsilon = 1e-10);
        assert_abs_diff_eq!(result[1], 3.2, epsilon = 1e-10);
    }

    #[test]
    fn test_lassa_basic() {
        // min (1/2n)||x - b||^2 + λ||x||_1, A=I, b=[3,-2], λ=1
        // Solution: soft_threshold(b, nλ) = soft_threshold([3,-2], 2) = [1, 0]
        let a = DMatrix::identity(2, 2);
        let b = DVector::from_vec(vec![3.0, -2.0]);
        let x0 = DVector::zeros(2);
        let pg = ProximalGradient::new(2000, 1e-8, 0.5);
        let result = pg.lasso(&a, &b, 1.0, &x0);
        assert_abs_diff_eq!(result.x[0], 1.0, epsilon = 0.05);
        assert_abs_diff_eq!(result.x[1], 0.0, epsilon = 0.05);
    }

    #[test]
    fn test_lassa_sparse() {
        // b = [0.1, 5.0], λ = 1, n = 2 → threshold = 2
        // soft_threshold([0.1, 5], 2) = [0, 3]
        let a = DMatrix::identity(2, 2);
        let b = DVector::from_vec(vec![0.1, 5.0]);
        let x0 = DVector::zeros(2);
        let pg = ProximalGradient::new(2000, 1e-8, 0.5);
        let result = pg.lasso(&a, &b, 1.0, &x0);
        assert_abs_diff_eq!(result.x[0], 0.0, epsilon = 0.05);
        assert_abs_diff_eq!(result.x[1], 3.0, epsilon = 0.05);
    }

    #[test]
    fn test_accelerated_lasso() {
        let a = DMatrix::identity(2, 2);
        let b = DVector::from_vec(vec![3.0, -2.0]);
        let x0 = DVector::zeros(2);
        let pg = ProximalGradient::accelerated(0.5);
        let result = pg.lasso(&a, &b, 1.0, &x0);
        assert_abs_diff_eq!(result.x[0], 1.0, epsilon = 0.05);
        assert_abs_diff_eq!(result.x[1], 0.0, epsilon = 0.05);
    }

    #[test]
    fn test_proximal_gradient_generic() {
        use crate::convex_functions::QuadraticFunction;
        let q = DMatrix::from_row_slice(2, 2, &[2.0, 0.0, 0.0, 2.0]);
        let c = DVector::from_vec(vec![-4.0, -6.0]);
        let f = QuadraticFunction::new(q, c);
        let pg = ProximalGradient::new(2000, 1e-8, 0.1);
        let result = pg.optimize(
            &f,
            &|x: &DVector<f64>| x.clone(), // Identity prox (no regularization)
            &DVector::from_vec(vec![0.0, 0.0]),
        );
        assert!(result.converged);
        assert_abs_diff_eq!(result.x[0], 2.0, epsilon = 0.1);
        assert_abs_diff_eq!(result.x[1], 3.0, epsilon = 0.1);
    }

    #[test]
    fn test_prox_quadratic() {
        let q = DMatrix::from_row_slice(2, 2, &[2.0, 0.0, 0.0, 2.0]);
        let x = DVector::from_vec(vec![1.0, 1.0]);
        let result = prox_quadratic(&x, &q, 1.0);
        // (I + Q)^{-1} x = diag(3, 3)^{-1} [1, 1] = [1/3, 1/3]
        assert_abs_diff_eq!(result[0], 1.0 / 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[1], 1.0 / 3.0, epsilon = 1e-10);
    }
}
