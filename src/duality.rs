//! Duality theory: Lagrangian, strong/weak duality, Slater's condition.

use crate::DVector;
use serde::{Serialize, Deserialize};

/// Type alias for scalar constraint functions.
pub type ConstraintFn = fn(&DVector<f64>) -> f64;
/// Type alias for constraint gradient functions.
pub type ConstraintGradFn = fn(&DVector<f64>) -> DVector<f64>;

/// Lagrangian for a problem: min f(x) s.t. g_i(x) <= 0, h_j(x) = 0.
/// L(x, λ, ν) = f(x) + Σ λ_i g_i(x) + Σ ν_j h_j(x)
pub struct Lagrangian {
    /// Primal objective f(x).
    pub objective: fn(&DVector<f64>) -> f64,
    /// Objective gradient.
    pub objective_grad: fn(&DVector<f64>) -> DVector<f64>,
    /// Inequality constraints g_i(x) <= 0.
    pub ineq_constraints: Vec<fn(&DVector<f64>) -> f64>,
    /// Inequality constraint gradients.
    pub ineq_grads: Vec<ConstraintGradFn>,
    /// Equality constraints h_j(x) = 0.
    pub eq_constraints: Vec<fn(&DVector<f64>) -> f64>,
    /// Equality constraint gradients.
    pub eq_grads: Vec<ConstraintGradFn>,
}

impl Lagrangian {
    /// Evaluate the Lagrangian at (x, λ, ν).
    pub fn eval(&self, x: &DVector<f64>, lambda: &DVector<f64>, nu: &DVector<f64>) -> f64 {
        let mut val = (self.objective)(x);
        for (i, g) in self.ineq_constraints.iter().enumerate() {
            val += lambda[i] * g(x);
        }
        for (j, h) in self.eq_constraints.iter().enumerate() {
            val += nu[j] * h(x);
        }
        val
    }

    /// Gradient of the Lagrangian w.r.t. x.
    pub fn gradient(&self, x: &DVector<f64>, lambda: &DVector<f64>, nu: &DVector<f64>) -> DVector<f64> {
        let mut grad = (self.objective_grad)(x);
        for (i, g_grad) in self.ineq_grads.iter().enumerate() {
            grad += g_grad(x) * lambda[i];
        }
        for (j, h_grad) in self.eq_grads.iter().enumerate() {
            grad += h_grad(x) * nu[j];
        }
        grad
    }

    /// Dual function: g(λ, ν) = inf_x L(x, λ, ν).
    /// We approximate by finding the x that makes ∇_x L = 0 (for convex problems).
    pub fn dual_function(
        &self,
        lambda: &DVector<f64>,
        nu: &DVector<f64>,
        x0: &DVector<f64>,
        steps: usize,
    ) -> f64 {
        // Approximate by gradient descent on L w.r.t. x
        let mut x = x0.clone();
        let lr = 0.01;
        for _ in 0..steps {
            let grad = self.gradient(&x, lambda, nu);
            x = &x - &grad * lr;
        }
        self.eval(&x, lambda, nu)
    }
}

/// Dual problem representation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DualProblem {
    /// Primal optimal value (if known).
    pub primal_optimal: Option<f64>,
    /// Dual optimal value (if known).
    pub dual_optimal: Option<f64>,
}

impl DualProblem {
    /// Compute duality gap.
    pub fn duality_gap(&self) -> Option<f64> {
        match (self.primal_optimal, self.dual_optimal) {
            (Some(p), Some(d)) => Some(p - d),
            _ => None,
        }
    }

    /// Check weak duality: p* >= d*.
    pub fn weak_duality_holds(&self) -> Option<bool> {
        match (self.primal_optimal, self.dual_optimal) {
            (Some(p), Some(d)) => Some(p >= d - 1e-10),
            _ => None,
        }
    }

    /// Check strong duality: p* == d*.
    pub fn strong_duality_holds(&self, tol: f64) -> Option<bool> {
        match (self.primal_optimal, self.dual_optimal) {
            (Some(p), Some(d)) => Some((p - d).abs() < tol),
            _ => None,
        }
    }
}

/// Slater's condition: there exists a strictly feasible x with g_i(x) < 0 for all i.
/// If Slater's condition holds and the problem is convex, strong duality holds.
pub fn check_slaters_condition<F1, F2>(
    ineq_constraints: &[F1],
    eq_constraints: &[F2],
    x: &DVector<f64>,
) -> bool
where
    F1: Fn(&DVector<f64>) -> f64,
    F2: Fn(&DVector<f64>) -> f64,
{
    // Check all inequality constraints strictly < 0
    for g in ineq_constraints {
        if g(x) >= 0.0 {
            return false;
        }
    }
    // Check all equality constraints = 0
    for h in eq_constraints {
        if (h(x)).abs() > 1e-10 {
            return false;
        }
    }
    true
}

/// Optimality certificate: verify KKT conditions.
pub struct OptimalityCertificate {
    pub stationarity_norm: f64,
    pub primal_feasible: bool,
    pub dual_feasible: bool,
    pub complementarity_gap: f64,
    pub is_optimal: bool,
}

impl OptimalityCertificate {
    pub fn check(
        lagrangian: &Lagrangian,
        x: &DVector<f64>,
        lambda: &DVector<f64>,
        nu: &DVector<f64>,
        tol: f64,
    ) -> Self {
        let grad = lagrangian.gradient(x, lambda, nu);
        let stationarity_norm = grad.norm();

        let primal_feasible = lagrangian
            .ineq_constraints
            .iter()
            .all(|g| g(x) <= tol)
            && lagrangian
                .eq_constraints
                .iter()
                .all(|h| (h(x)).abs() < tol);

        let dual_feasible = lambda.iter().all(|&l| l >= -tol);

        let complementarity_gap: f64 = lagrangian
            .ineq_constraints
            .iter()
            .enumerate()
            .map(|(i, g)| lambda[i] * g(x))
            .sum();

        let is_optimal = stationarity_norm < tol
            && primal_feasible
            && dual_feasible
            && complementarity_gap.abs() < tol;

        OptimalityCertificate {
            stationarity_norm,
            primal_feasible,
            dual_feasible,
            complementarity_gap,
            is_optimal,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn simple_lagrangian() -> Lagrangian {
        Lagrangian {
            objective: |x: &DVector<f64>| x[0] * x[0] + x[1] * x[1],
            objective_grad: |x: &DVector<f64>| DVector::from_vec(vec![2.0 * x[0], 2.0 * x[1]]),
            ineq_constraints: vec![
                |x: &DVector<f64>| x[0] + x[1] - 1.0,  // x1 + x2 <= 1
            ],
            ineq_grads: vec![
                |_: &DVector<f64>| DVector::from_vec(vec![1.0, 1.0]),
            ],
            eq_constraints: vec![],
            eq_grads: vec![],
        }
    }

    #[test]
    fn test_lagrangian_eval() {
        let lag = simple_lagrangian();
        let x = DVector::from_vec(vec![0.5, 0.5]);
        let lambda = DVector::from_vec(vec![1.0]);
        let nu = DVector::zeros(0);
        let val = lag.eval(&x, &lambda, &nu);
        // f = 0.5, λ*g = 1 * 0 = 0
        assert_abs_diff_eq!(val, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_lagrangian_gradient() {
        let lag = simple_lagrangian();
        let x = DVector::from_vec(vec![0.5, 0.5]);
        let lambda = DVector::from_vec(vec![2.0]);
        let nu = DVector::zeros(0);
        let grad = lag.gradient(&x, &lambda, &nu);
        // ∇f + λ∇g = [1, 1] + 2*[1, 1] = [3, 3]
        assert_abs_diff_eq!(grad[0], 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(grad[1], 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_weak_duality() {
        let dual = DualProblem {
            primal_optimal: Some(5.0),
            dual_optimal: Some(4.5),
        };
        assert_eq!(dual.weak_duality_holds(), Some(true));
        assert_abs_diff_eq!(dual.duality_gap().unwrap(), 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_strong_duality() {
        let dual = DualProblem {
            primal_optimal: Some(5.0),
            dual_optimal: Some(5.0),
        };
        assert_eq!(dual.strong_duality_holds(1e-8), Some(true));
    }

    #[test]
    fn test_slaters_condition_satisfied() {
        let ineq = vec![
            |x: &DVector<f64>| x[0] + x[1] - 1.0,
        ];
        let eq: Vec<fn(&DVector<f64>) -> f64> = vec![];
        let x = DVector::from_vec(vec![0.2, 0.3]);
        assert!(check_slaters_condition(&ineq, &eq, &x));
    }

    #[test]
    fn test_slaters_condition_violated() {
        let ineq = vec![
            |x: &DVector<f64>| x[0] + x[1] - 1.0,
        ];
        let eq: Vec<fn(&DVector<f64>) -> f64> = vec![];
        let x = DVector::from_vec(vec![0.6, 0.5]);
        assert!(!check_slaters_condition(&ineq, &eq, &x));
    }

    #[test]
    fn test_duality_gap_positive() {
        let dual = DualProblem {
            primal_optimal: Some(10.0),
            dual_optimal: Some(8.0),
        };
        assert!(dual.duality_gap().unwrap() > 0.0);
    }

    #[test]
    fn test_dual_function() {
        let lag = simple_lagrangian();
        let lambda = DVector::from_vec(vec![0.0]);
        let nu = DVector::zeros(0);
        let x0 = DVector::from_vec(vec![1.0, 1.0]);
        let dual_val = lag.dual_function(&lambda, &nu, &x0, 1000);
        // With λ=0, dual = inf of x1^2 + x2^2 = 0
        assert_abs_diff_eq!(dual_val, 0.0, epsilon = 0.1);
    }
}
