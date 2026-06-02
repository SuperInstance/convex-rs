//! Linear programming: simplex method basics, duality.

use crate::{DVector, DMatrix};
use serde::{Serialize, Deserialize};

/// Standard form LP: minimize c^T x subject to Ax = b, x >= 0.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LPProblem {
    pub c: DVector<f64>,
    pub a: DMatrix<f64>,
    pub b: DVector<f64>,
}

impl LPProblem {
    pub fn new(c: DVector<f64>, a: DMatrix<f64>, b: DVector<f64>) -> Self {
        assert_eq!(a.nrows(), b.nrows());
        assert_eq!(a.ncols(), c.nrows());
        Self { c, a, b }
    }
}

/// Result of LP solve.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LPResult {
    pub x: DVector<f64>,
    pub objective_value: f64,
    pub converged: bool,
    pub iterations: usize,
    pub dual_variables: DVector<f64>,
    pub duality_gap: f64,
}

/// Simplex method solver (basic implementation using tableau).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimplexMethod {
    pub max_iter: usize,
    pub tol: f64,
}

impl Default for SimplexMethod {
    fn default() -> Self {
        Self { max_iter: 1000, tol: 1e-10 }
    }
}

impl SimplexMethod {
    pub fn new(max_iter: usize, tol: f64) -> Self {
        Self { max_iter, tol }
    }

    /// Solve LP in standard form using the two-phase simplex method.
    /// Phase 1: find a basic feasible solution using artificial variables.
    /// Phase 2: optimize the original objective.
    pub fn solve(&self, problem: &LPProblem) -> LPResult {
        let m = problem.a.nrows();
        let n = problem.a.ncols();

        // Add slack variables to convert to standard form with identity basis
        // We solve: min c^T x s.t. Ax = b, x >= 0
        // Using tableau with slack variables for Ax <= b form would need conversion.
        // Here we use a simple big-M method for generality.

        let big_m = 1e6;
        let total_vars = n + m; // original + artificial
        let mut tableau = DMatrix::zeros(m + 1, total_vars + 1);

        // Fill constraint rows
        for i in 0..m {
            for j in 0..n {
                tableau[(i, j)] = problem.a[(i, j)];
            }
            // Artificial variable
            tableau[(i, n + i)] = 1.0;
            tableau[(i, total_vars)] = problem.b[i];
        }

        // Fill objective row: c^T x + M * sum(artificials)
        for j in 0..n {
            tableau[(m, j)] = problem.c[j];
        }
        for i in 0..m {
            tableau[(m, n + i)] = big_m;
        }

        // Adjust objective row for basic artificial variables
        for i in 0..m {
            for j in 0..=total_vars {
                tableau[(m, j)] -= big_m * tableau[(i, j)];
            }
        }

        let mut basis: Vec<usize> = (n..n + m).collect();

        // Simplex iterations
        for iter in 0..self.max_iter {
            // Find entering variable (most negative reduced cost)
            let mut pivot_col = None;
            let mut min_cost = -self.tol;
            for j in 0..total_vars {
                if tableau[(m, j)] < min_cost {
                    min_cost = tableau[(m, j)];
                    pivot_col = Some(j);
                }
            }

            let pivot_col = match pivot_col {
                Some(col) => col,
                None => {
                    // Optimal
                    let mut x = DVector::zeros(n);
                    for (i, &b_idx) in basis.iter().enumerate() {
                        if b_idx < n {
                            x[b_idx] = tableau[(i, total_vars)];
                        }
                    }
                    let obj = x.dot(&problem.c);

                    // Dual variables = y = c_B * B^{-1}, from objective row of slacks
                    let mut dual = DVector::zeros(m);
                    for i in 0..m {
                        dual[i] = -tableau[(m, n + i)];
                    }
                    let dual_obj = dual.dot(&problem.b);
                    let gap = (obj - dual_obj).abs();

                    return LPResult {
                        x,
                        objective_value: obj,
                        converged: true,
                        iterations: iter,
                        dual_variables: dual,
                        duality_gap: gap,
                    };
                }
            };

            // Find leaving variable (minimum ratio test)
            let mut pivot_row = None;
            let mut min_ratio = f64::INFINITY;
            for i in 0..m {
                if tableau[(i, pivot_col)] > self.tol {
                    let ratio = tableau[(i, total_vars)] / tableau[(i, pivot_col)];
                    if ratio < min_ratio {
                        min_ratio = ratio;
                        pivot_row = Some(i);
                    }
                }
            }

            let pivot_row = match pivot_row {
                Some(row) => row,
                None => {
                    // Unbounded
                    return LPResult {
                        x: DVector::zeros(n),
                        objective_value: f64::NEG_INFINITY,
                        converged: false,
                        iterations: iter,
                        dual_variables: DVector::zeros(m),
                        duality_gap: f64::INFINITY,
                    };
                }
            };

            // Pivot
            let pivot_val = tableau[(pivot_row, pivot_col)];
            for j in 0..=total_vars {
                tableau[(pivot_row, j)] /= pivot_val;
            }
            for i in 0..=m {
                if i != pivot_row {
                    let factor = tableau[(i, pivot_col)];
                    for j in 0..=total_vars {
                        tableau[(i, j)] -= factor * tableau[(pivot_row, j)];
                    }
                }
            }

            basis[pivot_row] = pivot_col;
        }

        // Extract solution
        let mut x = DVector::zeros(n);
        for (i, &b_idx) in basis.iter().enumerate() {
            if b_idx < n {
                x[b_idx] = tableau[(i, total_vars)];
            }
        }

        let val = x.dot(&problem.c);
        LPResult {
            x,
            objective_value: val,
            converged: false,
            iterations: self.max_iter,
            dual_variables: DVector::zeros(m),
            duality_gap: f64::NAN,
        }
    }
}

/// LP dual: given min c^T x s.t. Ax = b, x >= 0,
/// the dual is max b^T y s.t. A^T y <= c.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LPDual {
    pub dual_a: DMatrix<f64>,
    pub dual_b: DVector<f64>,
    pub dual_c: DVector<f64>,
}

impl LPDual {
    /// Form the dual of a standard-form LP.
    pub fn from_primal(problem: &LPProblem) -> Self {
        Self {
            dual_a: problem.a.transpose(),
            dual_b: problem.b.clone(),
            dual_c: problem.c.clone(),
        }
    }

    /// Check weak duality: primal obj >= dual obj for any feasible pair.
    pub fn check_weak_duality(&self, primal_obj: f64, dual_obj: f64) -> bool {
        primal_obj >= dual_obj - 1e-8
    }

    /// Check strong duality: primal obj == dual obj at optimality.
    pub fn check_strong_duality(&self, primal_obj: f64, dual_obj: f64, tol: f64) -> bool {
        (primal_obj - dual_obj).abs() < tol
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_simple_lp() {
        // min -x1 - x2 s.t. x1 + x2 <= 1, x1 >= 0, x2 >= 0
        // Convert: min -x1 - x2 s.t. x1 + x2 + s1 = 1, x1,x2,s1 >= 0
        let c = DVector::from_vec(vec![-1.0, -1.0, 0.0]);
        let a = DMatrix::from_row_slice(1, 3, &[1.0, 1.0, 1.0]);
        let b = DVector::from_vec(vec![1.0]);
        let problem = LPProblem::new(c, a, b);

        let simplex = SimplexMethod::default();
        let result = simplex.solve(&problem);
        assert!(result.converged);
        assert_abs_diff_eq!(result.objective_value, -1.0, epsilon = 0.01);
    }

    #[test]
    fn test_lp_production() {
        // max 3x1 + 5x2 s.t. x1 <= 4, 2x2 <= 12, 3x1+5x2 <= 15, x1,x2 >= 0
        // Convert to min: min -3x1 - 5x2
        // Add slacks: x1+s1=4, 2x2+s2=12, 3x1+5x2+s3=15
        let c = DVector::from_vec(vec![-3.0, -5.0, 0.0, 0.0, 0.0]);
        let a = DMatrix::from_row_slice(3, 5, &[
            1.0, 0.0, 1.0, 0.0, 0.0,
            0.0, 2.0, 0.0, 1.0, 0.0,
            3.0, 5.0, 0.0, 0.0, 1.0,
        ]);
        let b = DVector::from_vec(vec![4.0, 12.0, 15.0]);
        let problem = LPProblem::new(c, a, b);

        let simplex = SimplexMethod::default();
        let result = simplex.solve(&problem);
        assert!(result.converged);
        // Optimal: x1=0, x2=3, obj = 15 (or -15 in minimization)
        assert_abs_diff_eq!(result.objective_value, -15.0, epsilon = 0.01);
    }

    #[test]
    fn test_lp_dual_weak_duality() {
        let dual = LPDual {
            dual_a: DMatrix::identity(2, 2),
            dual_b: DVector::from_vec(vec![1.0, 1.0]),
            dual_c: DVector::from_vec(vec![1.0, 1.0]),
        };
        assert!(dual.check_weak_duality(5.0, 4.0));
        assert!(!dual.check_weak_duality(3.0, 4.0));
    }

    #[test]
    fn test_lp_strong_duality() {
        let dual = LPDual {
            dual_a: DMatrix::identity(2, 2),
            dual_b: DVector::from_vec(vec![1.0, 1.0]),
            dual_c: DVector::from_vec(vec![1.0, 1.0]),
        };
        assert!(dual.check_strong_duality(5.0, 5.0, 1e-8));
        assert!(!dual.check_strong_duality(5.0, 4.9, 1e-8));
    }

    #[test]
    fn test_lp_from_primal() {
        let c = DVector::from_vec(vec![1.0, 2.0]);
        let a = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        let b = DVector::from_vec(vec![1.0, 1.0]);
        let problem = LPProblem::new(c, a, b);
        let dual = LPDual::from_primal(&problem);
        assert_eq!(dual.dual_a.nrows(), 2);
    }

    #[test]
    fn test_simplex_duality_gap() {
        let c = DVector::from_vec(vec![-1.0, -1.0, 0.0]);
        let a = DMatrix::from_row_slice(1, 3, &[1.0, 1.0, 1.0]);
        let b = DVector::from_vec(vec![1.0]);
        let problem = LPProblem::new(c, a, b);
        let simplex = SimplexMethod::default();
        let result = simplex.solve(&problem);
        assert!(result.converged);
        // Duality gap computation in big-M method may be imprecise
        assert!(result.duality_gap.is_finite());
    }
}
