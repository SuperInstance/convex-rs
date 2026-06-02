//! Convex functions: verification, epigraphs, subgradients.

use crate::{DVector, DMatrix};
use serde::{Serialize, Deserialize};

/// Trait for objective functions used in optimization.
pub trait Objective: Send + Sync {
    /// Evaluate the function at x.
    fn eval(&self, x: &DVector<f64>) -> f64;
    /// Gradient at x.
    fn gradient(&self, x: &DVector<f64>) -> DVector<f64>;
    /// Hessian at x (optional, default panics).
    fn hessian(&self, _x: &DVector<f64>) -> DMatrix<f64> {
        unimplemented!("Hessian not available for this function")
    }
    /// Whether this function has a Hessian implementation.
    fn has_hessian(&self) -> bool {
        false
    }
}

/// Quadratic function: f(x) = 0.5 x^T Q x + c^T x
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuadraticFunction {
    pub q: DMatrix<f64>,
    pub c: DVector<f64>,
}

impl QuadraticFunction {
    pub fn new(q: DMatrix<f64>, c: DVector<f64>) -> Self {
        Self { q, c }
    }

    /// Check if Q is positive semidefinite (function is convex).
    pub fn is_convex(&self) -> bool {
        is_psd(&self.q)
    }

    /// Check if Q is positive definite (function is strictly convex).
    pub fn is_strictly_convex(&self) -> bool {
        is_pd(&self.q)
    }

    /// Minimum value (assuming convex) at x* = -Q^{-1} c.
    pub fn minimum(&self) -> Option<(DVector<f64>, f64)> {
        if !is_pd(&self.q) {
            return None;
        }
        let qr = self.q.clone().qr();
        let x_star = qr.solve(&-&self.c)?;
        let val = self.eval(&x_star);
        Some((x_star, val))
    }
}

impl Objective for QuadraticFunction {
    fn eval(&self, x: &DVector<f64>) -> f64 {
        0.5 * x.dot(&(&self.q * x)) + self.c.dot(x)
    }

    fn gradient(&self, x: &DVector<f64>) -> DVector<f64> {
        &self.q * x + &self.c
    }

    fn hessian(&self, _x: &DVector<f64>) -> DMatrix<f64> {
        self.q.clone()
    }

    fn has_hessian(&self) -> bool {
        true
    }
}

/// Check if a matrix is positive semidefinite.
pub fn is_psd(m: &DMatrix<f64>) -> bool {
    let symmetric = (m + m.transpose()) * 0.5;
    let eig = symmetric.symmetric_eigen();
    eig.eigenvalues.iter().all(|&v| v >= -1e-10)
}

/// Check if a matrix is positive definite.
pub fn is_pd(m: &DMatrix<f64>) -> bool {
    let symmetric = (m + m.transpose()) * 0.5;
    let eig = symmetric.symmetric_eigen();
    eig.eigenvalues.iter().all(|&v| v > 1e-10)
}

/// Verify convexity of a function along a line segment.
/// A function is convex iff f(θx + (1-θ)y) <= θf(x) + (1-θ)f(y) for all θ ∈ [0,1].
pub fn verify_convexity_line(
    f: &dyn Objective,
    x: &DVector<f64>,
    y: &DVector<f64>,
    n_samples: usize,
    tol: f64,
) -> bool {
    for i in 0..=n_samples {
        let theta = i as f64 / n_samples as f64;
        let mid = x * (1.0 - theta) + y * theta;
        let f_mid = f.eval(&mid);
        let f_line = (1.0 - theta) * f.eval(x) + theta * f.eval(y);
        if f_mid > f_line + tol {
            return false;
        }
    }
    true
}

/// Epigraph of a function: epi(f) = {(x, t) | f(x) <= t}.
/// A function is convex iff its epigraph is a convex set.
pub fn epigraph_check(
    f: &dyn Objective,
    x: &DVector<f64>,
    t_x: f64,
    y: &DVector<f64>,
    t_y: f64,
    theta: f64,
) -> bool {
    let mid = x * (1.0 - theta) + y * theta;
    let t_mid = (1.0 - theta) * t_x + theta * t_y;
    f.eval(&mid) <= t_mid + 1e-10
}

/// Subgradient oracle: returns a subgradient at x.
/// For differentiable convex functions, this is just the gradient.
pub struct SubgradientOracle<F: Objective> {
    pub f: F,
}

impl<F: Objective> SubgradientOracle<F> {
    pub fn new(f: F) -> Self {
        Self { f }
    }

    pub fn subgradient(&self, x: &DVector<f64>) -> DVector<f64> {
        self.f.gradient(x)
    }

    pub fn eval(&self, x: &DVector<f64>) -> f64 {
        self.f.eval(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn make_convex_quadratic() -> QuadraticFunction {
        let q = DMatrix::from_row_slice(2, 2, &[
            2.0, 0.0,
            0.0, 4.0,
        ]);
        let c = DVector::from_vec(vec![0.0, 0.0]);
        QuadraticFunction::new(q, c)
    }

    #[test]
    fn test_quadratic_eval() {
        let f = make_convex_quadratic();
        let x = DVector::from_vec(vec![1.0, 1.0]);
        assert_abs_diff_eq!(f.eval(&x), 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_quadratic_gradient() {
        let f = make_convex_quadratic();
        let x = DVector::from_vec(vec![1.0, 1.0]);
        let g = f.gradient(&x);
        assert_abs_diff_eq!(g[0], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(g[1], 4.0, epsilon = 1e-10);
    }

    #[test]
    fn test_quadratic_hessian() {
        let f = make_convex_quadratic();
        let x = DVector::from_vec(vec![1.0, 1.0]);
        let h = f.hessian(&x);
        assert_abs_diff_eq!(h[(0, 0)], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(h[(1, 1)], 4.0, epsilon = 1e-10);
    }

    #[test]
    fn test_quadratic_is_convex() {
        let f = make_convex_quadratic();
        assert!(f.is_convex());
        assert!(f.is_strictly_convex());
    }

    #[test]
    fn test_quadratic_minimum() {
        let f = make_convex_quadratic();
        let (x_star, val) = f.minimum().unwrap();
        assert_abs_diff_eq!(x_star[0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(x_star[1], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(val, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_non_convex_quadratic() {
        let q = DMatrix::from_row_slice(2, 2, &[
            -2.0, 0.0,
            0.0, 1.0,
        ]);
        let c = DVector::from_vec(vec![0.0, 0.0]);
        let f = QuadraticFunction::new(q, c);
        assert!(!f.is_convex());
    }

    #[test]
    fn test_convexity_line_verification() {
        let f = make_convex_quadratic();
        let x = DVector::from_vec(vec![1.0, 2.0]);
        let y = DVector::from_vec(vec![3.0, -1.0]);
        assert!(verify_convexity_line(&f, &x, &y, 50, 1e-8));
    }

    #[test]
    fn test_epigraph_check() {
        let f = make_convex_quadratic();
        let x = DVector::from_vec(vec![1.0, 0.0]);
        let y = DVector::from_vec(vec![0.0, 1.0]);
        let t_x = f.eval(&x) + 1.0; // Above epigraph
        let t_y = f.eval(&y) + 1.0;
        assert!(epigraph_check(&f, &x, t_x, &y, t_y, 0.5));
    }

    #[test]
    fn test_psd_identity() {
        let i = DMatrix::identity(3, 3);
        assert!(is_psd(&i));
        assert!(is_pd(&i));
    }

    #[test]
    fn test_subgradient_oracle() {
        let f = make_convex_quadratic();
        let oracle = SubgradientOracle::new(f);
        let x = DVector::from_vec(vec![1.0, 1.0]);
        let g = oracle.subgradient(&x);
        assert_abs_diff_eq!(g[0], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(g[1], 4.0, epsilon = 1e-10);
    }
}
