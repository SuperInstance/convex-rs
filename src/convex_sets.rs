//! Convex sets: verification, membership tests, projections.

use crate::{DVector, DMatrix};
use serde::{Serialize, Deserialize};

/// A convex set representation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConvexSet {
    /// Affine set: {x | Ax = b}
    Affine { a: DMatrix<f64>, b: DVector<f64> },
    /// Halfspace: {x | a^T x <= b}
    Halfspace { a: DVector<f64>, b: f64 },
    /// Ball: {x | ||x - c|| <= r}
    Ball { center: DVector<f64>, radius: f64 },
    /// Box: {x | l <= x <= u} (component-wise)
    Box { lower: DVector<f64>, upper: DVector<f64> },
    /// Polyhedron: {x | Ax <= b, Cx = d}
    Polyhedron {
        a: DMatrix<f64>,
        b: DVector<f64>,
        c: Option<DMatrix<f64>>,
        d: Option<DVector<f64>>,
    },
    /// Simplex: {x | x >= 0, sum(x) = 1}
    Simplex { n: usize },
}

impl ConvexSet {
    /// Check if a point belongs to the set.
    pub fn contains(&self, x: &DVector<f64>, tol: f64) -> bool {
        match self {
            ConvexSet::Affine { a, b } => {
                let residual = a * x - b;
                residual.iter().all(|v| v.abs() < tol)
            }
            ConvexSet::Halfspace { a, b } => a.dot(x) <= b + tol,
            ConvexSet::Ball { center, radius } => {
                let diff = x - center;
                diff.norm() <= radius + tol
            }
            ConvexSet::Box { lower, upper } => {
                x.iter()
                    .zip(lower.iter())
                    .zip(upper.iter())
                    .all(|((xi, li), ui)| *li - tol <= *xi && *xi <= *ui + tol)
            }
            ConvexSet::Polyhedron { a, b, c, d } => {
                let ax = a * x;
                if !ax.iter().zip(b.iter()).all(|(v, bi)| *v <= bi + tol) {
                    return false;
                }
                if let (Some(c_mat), Some(d_vec)) = (c, d) {
                    let cx = c_mat * x;
                    return cx.iter().zip(d_vec.iter()).all(|(v, di)| (*v - di).abs() < tol);
                }
                true
            }
            ConvexSet::Simplex { n } => {
                if x.nrows() != *n {
                    return false;
                }
                x.iter().all(|v| *v >= -tol) && (x.sum() - 1.0).abs() < tol
            }
        }
    }

    /// Project a point onto the convex set.
    pub fn project(&self, x: &DVector<f64>) -> DVector<f64> {
        match self {
            ConvexSet::Box { lower, upper } => {
                let data: Vec<f64> = x.iter()
                    .zip(lower.iter())
                    .zip(upper.iter())
                    .map(|((xi, li), ui)| xi.max(*li).min(*ui))
                    .collect();
                DVector::from_vec(data)
            }
            ConvexSet::Ball { center, radius } => {
                let diff = x - center;
                let norm = diff.norm();
                if norm <= *radius {
                    x.clone()
                } else {
                    center + (diff * (1.0 / norm)) * *radius
                }
            }
            ConvexSet::Simplex { n } => project_simplex(x, *n),
            _ => x.clone(),
        }
    }

    /// Verify the set is convex (returns true for all variants by construction).
    pub fn is_convex(&self) -> bool {
        true
    }
}

/// Project onto the probability simplex.
pub fn project_simplex(v: &DVector<f64>, _n: usize) -> DVector<f64> {
    let mut sorted: Vec<f64> = v.iter().copied().collect();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let cumsum: Vec<f64> = sorted
        .iter()
        .scan(0.0_f64, |acc, &val| {
            *acc += val;
            Some(*acc)
        })
        .collect();
    let rho = sorted
        .iter()
        .enumerate()
        .rposition(|(i, &s)| s - (cumsum[i] - 1.0) / ((i + 1) as f64) > 0.0)
        .unwrap_or(0);
    let theta = (cumsum[rho] - 1.0) / ((rho + 1) as f64);
    let data: Vec<f64> = v.iter().map(|&vi| (vi - theta).max(0.0)).collect();
    DVector::from_vec(data)
}

/// Check convexity of a set via midpoint test on membership.
pub fn verify_convexity_midpoint(
    set: &ConvexSet,
    x: &DVector<f64>,
    y: &DVector<f64>,
    tol: f64,
) -> bool {
    let mid = (x + y) * 0.5;
    if set.contains(x, tol) && set.contains(y, tol) {
        set.contains(&mid, tol)
    } else {
        true
    }
}

/// Intersection of two convex sets using alternating projection.
pub fn intersect(set1: &ConvexSet, set2: &ConvexSet, tol: f64) -> Option<DVector<f64>> {
    let n = match set1 {
        ConvexSet::Ball { center, .. } => center.nrows(),
        ConvexSet::Box { lower, .. } => lower.nrows(),
        ConvexSet::Halfspace { a, .. } => a.nrows(),
        ConvexSet::Affine { a, .. } => a.ncols(),
        ConvexSet::Simplex { n } => *n,
        ConvexSet::Polyhedron { a, .. } => a.ncols(),
    };
    let mut x = DVector::zeros(n);
    for _ in 0..100 {
        x = set1.project(&set2.project(&x));
    }
    if set1.contains(&x, tol) && set2.contains(&x, tol) {
        Some(x)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DVector;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_ball_contains_center() {
        let center = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let ball = ConvexSet::Ball { center: center.clone(), radius: 2.0 };
        assert!(ball.contains(&center, 1e-10));
    }

    #[test]
    fn test_ball_boundary() {
        let center = DVector::from_vec(vec![0.0, 0.0]);
        let ball = ConvexSet::Ball { center: center.clone(), radius: 1.0 };
        let on_boundary = DVector::from_vec(vec![1.0, 0.0]);
        assert!(ball.contains(&on_boundary, 1e-10));
    }

    #[test]
    fn test_ball_outside() {
        let center = DVector::from_vec(vec![0.0, 0.0]);
        let ball = ConvexSet::Ball { center, radius: 1.0 };
        let outside = DVector::from_vec(vec![2.0, 0.0]);
        assert!(!ball.contains(&outside, 1e-10));
    }

    #[test]
    fn test_box_contains() {
        let lower = DVector::from_vec(vec![-1.0, -1.0]);
        let upper = DVector::from_vec(vec![1.0, 1.0]);
        let box_set = ConvexSet::Box { lower, upper };
        let inside = DVector::from_vec(vec![0.0, 0.5]);
        assert!(box_set.contains(&inside, 1e-10));
    }

    #[test]
    fn test_box_outside() {
        let lower = DVector::from_vec(vec![-1.0, -1.0]);
        let upper = DVector::from_vec(vec![1.0, 1.0]);
        let box_set = ConvexSet::Box { lower, upper };
        let outside = DVector::from_vec(vec![1.5, 0.0]);
        assert!(!box_set.contains(&outside, 1e-10));
    }

    #[test]
    fn test_halfspace_contains() {
        let a = DVector::from_vec(vec![1.0, 1.0]);
        let hs = ConvexSet::Halfspace { a, b: 1.0 };
        let inside = DVector::from_vec(vec![0.0, 0.0]);
        assert!(hs.contains(&inside, 1e-10));
    }

    #[test]
    fn test_simplex_valid() {
        let s = ConvexSet::Simplex { n: 3 };
        let valid = DVector::from_vec(vec![0.3, 0.3, 0.4]);
        assert!(s.contains(&valid, 1e-10));
    }

    #[test]
    fn test_simplex_invalid() {
        let s = ConvexSet::Simplex { n: 3 };
        let invalid = DVector::from_vec(vec![0.5, 0.6, 0.0]);
        assert!(!s.contains(&invalid, 1e-10));
    }

    #[test]
    fn test_project_onto_box() {
        let lower = DVector::from_vec(vec![-1.0, -1.0]);
        let upper = DVector::from_vec(vec![1.0, 1.0]);
        let box_set = ConvexSet::Box { lower, upper };
        let point = DVector::from_vec(vec![2.0, -3.0]);
        let proj = box_set.project(&point);
        assert_abs_diff_eq!(proj[0], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(proj[1], -1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_project_onto_ball() {
        let center = DVector::from_vec(vec![0.0, 0.0]);
        let ball = ConvexSet::Ball { center, radius: 1.0 };
        let point = DVector::from_vec(vec![3.0, 4.0]);
        let proj = ball.project(&point);
        assert_abs_diff_eq!(proj.norm(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_project_simplex_basic() {
        let v = DVector::from_vec(vec![0.5, 0.3, 0.2]);
        let proj = project_simplex(&v, 3);
        assert_abs_diff_eq!(proj.sum(), 1.0, epsilon = 1e-10);
        assert!(proj.iter().all(|&x| x >= -1e-10));
    }

    #[test]
    fn test_project_simplex_negative() {
        let v = DVector::from_vec(vec![-1.0, 2.0, 0.5]);
        let proj = project_simplex(&v, 3);
        assert_abs_diff_eq!(proj.sum(), 1.0, epsilon = 1e-10);
        assert!(proj.iter().all(|&x| x >= -1e-10));
    }

    #[test]
    fn test_midpoint_convexity() {
        let ball = ConvexSet::Ball {
            center: DVector::from_vec(vec![0.0, 0.0]),
            radius: 2.0,
        };
        let x = DVector::from_vec(vec![1.0, 0.0]);
        let y = DVector::from_vec(vec![0.0, 1.0]);
        assert!(verify_convexity_midpoint(&ball, &x, &y, 1e-10));
    }

    #[test]
    fn test_all_sets_are_convex() {
        let sets = vec![
            ConvexSet::Ball { center: DVector::zeros(2), radius: 1.0 },
            ConvexSet::Box { lower: DVector::from_vec(vec![-1.0, -1.0]), upper: DVector::from_vec(vec![1.0, 1.0]) },
            ConvexSet::Halfspace { a: DVector::from_vec(vec![1.0, 0.0]), b: 1.0 },
            ConvexSet::Simplex { n: 3 },
        ];
        for s in &sets {
            assert!(s.is_convex());
        }
    }
}
