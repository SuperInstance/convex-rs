//! # convex-rs
//!
//! Convex optimization library: gradient descent, Newton's method, interior point methods,
//! linear programming, quadratic programming, duality theory, proximal operators, and
//! resource allocation.
//!
//! Every local minimum is a global minimum.

#![deny(unsafe_code)]

pub mod convex_sets;
pub mod convex_functions;
pub mod gradient_descent;
pub mod newton;
pub mod interior_point;
pub mod linear_programming;
pub mod quadratic_programming;
pub mod duality;
pub mod proximal;
pub mod resource_allocation;

pub use nalgebra::{DVector, DMatrix};
