//! Quadratic programming: KKT conditions, active set method.

use crate::{DVector, DMatrix};
use serde::{Serialize, Deserialize};

/// Quadratic program: minimize 0.5 x^T Q x + c^T x subject to Ax <= b, Ex = d.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QPProblem {
    pub q: DMatrix<f64>,
    pub c: DVector<f64>,
    pub a: DMatrix<f64>,  // Inequality constraints: Ax <= b
    pub b: DVector<f64>,
    pub e: Option<DMatrix<f64>>,  // Equality constraints: Ex = d
    pub d: Option<DVector<f64>>,
}

impl QPProblem {
    pub fn new(q: DMatrix<f64>, c: DVector<f64>, a: DMatrix<f64>, b: DVector<f64>) -> Self {
        Self { q, c, a, b, e: None, d: None }
    }

    pub fn with_equality(mut self, e: DMatrix<f64>, d: DVector<f64>) -> Self {
        self.e = Some(e.clone());
        self.d = Some(d.clone());
        Self { q: self.q, c: self.c, a: self.a, b: self.b, e: Some(e), d: Some(d) }
    }
}

/// Result of QP solve.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QPResult {
    pub x: DVector<f64>,
    pub objective_value: f64,
    pub converged: bool,
    pub iterations: usize,
    pub lambda: DVector<f64>,  // Dual variables for inequalities
    pub nu: Option<DVector<f64>>,  // Dual variables for equalities
}

/// KKT conditions for a QP.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KKTConditions {
    pub stationarity: DVector<f64>,
    pub primal_feasibility_ineq: DVector<f64>,
    pub primal_feasibility_eq: Option<DVector<f64>>,
    pub dual_feasibility: DVector<f64>,
    pub complementarity: f64,
    pub satisfied: bool,
}

impl KKTConditions {
    /// Check KKT conditions at a given point.
    pub fn check(
        problem: &QPProblem,
        x: &DVector<f64>,
        lambda: &DVector<f64>,
        nu: Option<&DVector<f64>>,
        tol: f64,
    ) -> Self {
        let _n = x.nrows();
        let _m = lambda.nrows();

        // Stationarity: Qx + c + A^T λ + E^T ν = 0
        let mut stationarity = &problem.q * x + &problem.c + &problem.a.transpose() * lambda;
        if let (Some(ref e), Some(nu_vec)) = (&problem.e, nu) {
            stationarity += &(e.transpose()) * nu_vec;
        }

        // Primal feasibility (inequality): Ax - b <= 0
        let primal_ineq = &problem.a * x - &problem.b;

        // Primal feasibility (equality): Ex - d = 0
        let primal_eq = match (&problem.e, &problem.d) {
            (Some(e), Some(d)) => Some(e * x - d),
            _ => None,
        };

        // Dual feasibility: λ >= 0
        let dual_feas = lambda.clone();

        // Complementarity: λ_i * (Ax - b)_i = 0
        let comp_slack: f64 = lambda
            .iter()
            .zip(primal_ineq.iter())
            .map(|(&lam, &slack)| lam * slack)
            .sum();

        let satisfied = stationarity.norm() < tol
            && primal_ineq.iter().all(|v| *v <= tol)
            && lambda.iter().all(|v| *v >= -tol)
            && comp_slack.abs() < tol
            && primal_eq
                .as_ref()
                .is_none_or(|eq| eq.norm() < tol);

        KKTConditions {
            stationarity,
            primal_feasibility_ineq: primal_ineq,
            primal_feasibility_eq: primal_eq,
            dual_feasibility: dual_feas,
            complementarity: comp_slack,
            satisfied,
        }
    }
}

/// Active set method for QP.
#[derive(Debug, Clone)]
pub struct ActiveSetQP {
    pub max_iter: usize,
    pub tol: f64,
}

impl Default for ActiveSetQP {
    fn default() -> Self {
        Self { max_iter: 200, tol: 1e-8 }
    }
}

impl ActiveSetQP {
    pub fn new(max_iter: usize, tol: f64) -> Self {
        Self { max_iter, tol }
    }

    /// Solve QP using active set method.
    pub fn solve(&self, problem: &QPProblem) -> QPResult {
        let n = problem.q.nrows();
        let m = problem.a.nrows();

        let mut x = DVector::zeros(n);
        let mut lambda = DVector::zeros(m);
        let mut active_set: Vec<usize> = Vec::new();

        let q_inv = problem.q.clone().try_inverse();

        for iter in 0..self.max_iter {
            let grad = &problem.q * &x + &problem.c;

            let p = if active_set.is_empty() {
                match &q_inv {
                    Some(qi) => -(qi * &grad),
                    None => &grad * (-0.01),
                }
            } else {
                let n_active = active_set.len();
                let mut kkt_matrix = DMatrix::zeros(n + n_active, n + n_active);
                let mut kkt_rhs = DVector::zeros(n + n_active);

                for i in 0..n {
                    for j in 0..n {
                        kkt_matrix[(i, j)] = problem.q[(i, j)];
                    }
                    kkt_rhs[i] = -grad[i];
                }
                for (idx, &ai) in active_set.iter().enumerate() {
                    for j in 0..n {
                        kkt_matrix[(n + idx, j)] = problem.a[(ai, j)];
                        kkt_matrix[(j, n + idx)] = problem.a[(ai, j)];
                    }
                }

                match kkt_matrix.try_inverse() {
                    Some(inv) => {
                        let sol = inv * kkt_rhs;
                        sol.rows(0, n).into_owned()
                    }
                    None => DVector::zeros(n),
                }
            };

            // Update dual variables
            if let Some(ref qi) = q_inv {
                let at_lambda = &problem.a.transpose() * &lambda;
                let combined = &grad + &at_lambda;
                for &ai in &active_set {
                    let a_row = problem.a.row(ai);
                    let val: f64 = (a_row * qi * &combined)[0];
                    lambda[ai] = val;
                }
            }

            if p.norm() < self.tol {
                let all_nonneg = active_set.iter().all(|&ai| lambda[ai] >= -self.tol);
                if all_nonneg {
                    let qx = &problem.q * &x;
                    let obj = 0.5 * x.dot(&qx) + problem.c.dot(&x);
                    return QPResult {
                        x: x.clone(),
                        objective_value: obj,
                        converged: true,
                        iterations: iter,
                        lambda: lambda.clone(),
                        nu: None,
                    };
                }
                if let Some(pos) = active_set
                    .iter()
                    .enumerate()
                    .filter(|(_, &ai)| lambda[ai] < -self.tol)
                    .min_by(|(_, a), (_, b)| lambda[**a].partial_cmp(&lambda[**b]).unwrap())
                    .map(|(pos, _)| pos)
                {
                    active_set.remove(pos);
                }
            } else {
                let mut alpha = 1.0_f64;
                let mut blocking = None;
                for i in 0..m {
                    if active_set.contains(&i) {
                        continue;
                    }
                    let ap: f64 = (0..n).map(|j| problem.a[(i, j)] * p[j]).sum();
                    if ap > self.tol {
                        let ax: f64 = (0..n).map(|j| problem.a[(i, j)] * x[j]).sum();
                        let slack = problem.b[i] - ax;
                        let ratio = slack / ap;
                        if ratio < alpha {
                            alpha = ratio;
                            blocking = Some(i);
                        }
                    }
                }

                x += &p * alpha;

                if let Some(bi) = blocking {
                    active_set.push(bi);
                }
            }
        }

        let qx = &problem.q * &x;
        let obj = 0.5 * x.dot(&qx) + problem.c.dot(&x);
        QPResult {
            x,
            objective_value: obj,
            converged: false,
            iterations: self.max_iter,
            lambda,
            nu: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_unconstrained_qp() {
        let q = DMatrix::from_row_slice(2, 2, &[2.0, 0.0, 0.0, 2.0]);
        let c = DVector::from_vec(vec![-4.0, -6.0]);
        let a = DMatrix::zeros(0, 2);
        let b = DVector::zeros(0);
        let problem = QPProblem::new(q, c, a, b);
        let solver = ActiveSetQP::default();
        let result = solver.solve(&problem);
        assert!(result.converged);
        assert_abs_diff_eq!(result.x[0], 2.0, epsilon = 0.01);
        assert_abs_diff_eq!(result.x[1], 3.0, epsilon = 0.01);
    }

    #[test]
    fn test_constrained_qp() {
        let q = DMatrix::from_row_slice(2, 2, &[2.0, 0.0, 0.0, 2.0]);
        let c = DVector::from_vec(vec![-4.0, -6.0]);
        // x1 + x2 <= 2
        let a = DMatrix::from_row_slice(1, 2, &[1.0, 1.0]);
        let b = DVector::from_vec(vec![2.0]);
        let problem = QPProblem::new(q, c, a, b);
        let solver = ActiveSetQP::new(500, 1e-8);
        let result = solver.solve(&problem);
        // The optimal solution is x = (0.5, 1.5) with λ = 3
        // obj = 0.5*0.5 + 0.5*4.5 - 2 - 9 = -8.25
        assert!(result.objective_value < -7.0, "obj = {}", result.objective_value);
        assert_abs_diff_eq!(result.x[0] + result.x[1], 2.0, epsilon = 0.2);
    }

    #[test]
    fn test_kkt_conditions_unconstrained() {
        let q = DMatrix::from_row_slice(2, 2, &[2.0, 0.0, 0.0, 2.0]);
        let c = DVector::from_vec(vec![-4.0, -6.0]);
        let a = DMatrix::zeros(0, 2);
        let b = DVector::zeros(0);
        let problem = QPProblem::new(q, c, a, b);
        let x = DVector::from_vec(vec![2.0, 3.0]);
        let lambda = DVector::zeros(0);
        let kkt = KKTConditions::check(&problem, &x, &lambda, None, 1e-8);
        assert!(kkt.satisfied);
    }

    #[test]
    fn test_kkt_conditions_constrained() {
        let q = DMatrix::from_row_slice(2, 2, &[2.0, 0.0, 0.0, 2.0]);
        let c = DVector::from_vec(vec![-4.0, -6.0]);
        let a = DMatrix::from_row_slice(1, 2, &[1.0, 1.0]);
        let b = DVector::from_vec(vec![2.0]);
        let problem = QPProblem::new(q, c, a, b);

        // Unconstrained opt: (2,3), but x1+x2=5 > 2, so constraint active
        // With active constraint: x = (0.5, 1.5), λ = 2
        let x = DVector::from_vec(vec![0.5, 1.5]);
        let lambda = DVector::from_vec(vec![2.0]);
        let _kkt = KKTConditions::check(&problem, &x, &lambda, None, 1e-6);
        // Qx + c + A^T λ = [1, 3] + [-4, -6] + [2, 2] = [-1, -1]... not zero
        // Let me compute properly: Qx = [1, 3], c = [-4, -6], Qx+c = [-3, -3]
        // A^T λ = [2, 2], Qx+c+A^Tλ = [-1, -1] ≠ 0, so not satisfied
        // Correct: x = (0, 2), λ = 2: Qx+c+A^Tλ = [0,4]+[-4,-6]+[2,2] = [-2,0] ≠ 0
        // Actually, x = (1, 1), λ = 2: Qx+c+A^Tλ = [2,2]+[-4,-6]+[2,2] = [0,-2] ≠ 0
        // Correct solution: min 0.5*2x1^2+0.5*2x2^2-4x1-6x2 s.t. x1+x2=2
        // Lagrangian: L = x1^2+x2^2-4x1-6x2+λ(x1+x2-2)
        // dL/dx1 = 2x1-4+λ = 0, dL/dx2 = 2x2-6+λ = 0
        // x1 = (4-λ)/2, x2 = (6-λ)/2
        // x1+x2 = (10-2λ)/2 = 5-λ = 2, so λ = 3
        // x1 = 0.5, x2 = 1.5
        let x = DVector::from_vec(vec![0.5, 1.5]);
        let lambda = DVector::from_vec(vec![3.0]);
        let kkt = KKTConditions::check(&problem, &x, &lambda, None, 1e-6);
        assert!(kkt.satisfied);
    }

    #[test]
    fn test_qp_box_constraints() {
        // min (x-1)^2 + (y-1)^2 s.t. 0 <= x <= 0.5, 0 <= y <= 0.5
        let q = DMatrix::from_row_slice(2, 2, &[2.0, 0.0, 0.0, 2.0]);
        let c = DVector::from_vec(vec![-2.0, -2.0]);
        let a = DMatrix::from_row_slice(4, 2, &[
            1.0, 0.0,
            0.0, 1.0,
            -1.0, 0.0,
            0.0, -1.0,
        ]);
        let b = DVector::from_vec(vec![0.5, 0.5, 0.0, 0.0]);
        let problem = QPProblem::new(q, c, a, b);
        let solver = ActiveSetQP::new(500, 1e-8);
        let result = solver.solve(&problem);
        // Optimal: x = (0.5, 0.5)
        assert!(result.x[0] >= 0.4 && result.x[0] <= 0.6, "x[0] = {}", result.x[0]);
        assert!(result.x[1] >= 0.4 && result.x[1] <= 0.6, "x[1] = {}", result.x[1]);
    }
}
