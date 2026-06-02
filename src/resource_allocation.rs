//! Resource allocation: optimal distribution under constraints.

use crate::{DVector, DMatrix};
use crate::convex_functions::Objective;
use crate::gradient_descent::{GradientDescent, StepSize};
use serde::{Serialize, Deserialize};

/// A resource allocation agent with utility function and budget constraint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Agent {
    pub id: usize,
    /// Utility weights for each resource type.
    pub utility_weights: DVector<f64>,
    /// Risk aversion parameter (higher = more risk averse).
    pub risk_aversion: f64,
    /// Budget constraint (total allocation <= budget).
    pub budget: f64,
}

impl Agent {
    pub fn new(id: usize, utility_weights: DVector<f64>, risk_aversion: f64, budget: f64) -> Self {
        Self { id, utility_weights, risk_aversion, budget }
    }

    /// Compute utility for a given resource allocation.
    /// U(x) = w^T x - (α/2) ||x||^2 (quadratic utility with risk aversion).
    pub fn utility(&self, x: &DVector<f64>) -> f64 {
        self.utility_weights.dot(x) - 0.5 * self.risk_aversion * x.norm_squared()
    }

    /// Gradient of utility.
    pub fn utility_gradient(&self, x: &DVector<f64>) -> DVector<f64> {
        &self.utility_weights - x * self.risk_aversion
    }

    /// Hessian of utility.
    pub fn utility_hessian(&self, n: usize) -> DMatrix<f64> {
        DMatrix::from_diagonal_element(n, n, -self.risk_aversion)
    }
}

/// Resource allocation problem.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceAllocation {
    /// Agents competing for resources.
    pub agents: Vec<Agent>,
    /// Total available resources.
    pub total_resources: DVector<f64>,
    /// Number of resource types.
    pub n_resources: usize,
}

/// Result of resource allocation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationResult {
    /// Allocation for each agent: allocations[i] is the vector for agent i.
    pub allocations: Vec<DVector<f64>>,
    /// Total social welfare (sum of utilities).
    pub social_welfare: f64,
    /// Individual utilities.
    pub utilities: Vec<f64>,
    /// Whether constraints are satisfied.
    pub feasible: bool,
    /// Convergence status.
    pub converged: bool,
}

/// Utility for the social welfare optimization problem.
struct SocialWelfareObjective {
    agents: Vec<Agent>,
    n_resources: usize,
}

impl Objective for SocialWelfareObjective {
    fn eval(&self, x: &DVector<f64>) -> f64 {
        let n_agents = self.agents.len();
        let mut total = 0.0;
        for i in 0..n_agents {
            let alloc = x.rows(i * self.n_resources, self.n_resources);
            total -= self.agents[i].utility(&alloc.into_owned());
        }
        total
    }

    fn gradient(&self, x: &DVector<f64>) -> DVector<f64> {
        let n_agents = self.agents.len();
        let mut grad = DVector::zeros(x.nrows());
        for i in 0..n_agents {
            let alloc = x.rows(i * self.n_resources, self.n_resources).into_owned();
            let g = self.agents[i].utility_gradient(&alloc);
            for j in 0..self.n_resources {
                grad[i * self.n_resources + j] = -g[j];
            }
        }
        grad
    }
}

impl ResourceAllocation {
    pub fn new(agents: Vec<Agent>, total_resources: DVector<f64>) -> Self {
        let n_resources = total_resources.nrows();
        Self { agents, total_resources, n_resources }
    }

    /// Solve: maximize social welfare = Σ U_i(x_i) subject to:
    /// - Σ x_i <= total_resources (resource constraints)
    /// - x_i >= 0 (non-negativity)
    /// - Σ_j x_{i,j} <= budget_i (budget constraints)
    pub fn allocate(&self) -> AllocationResult {
        let n_agents = self.agents.len();
        let n_res = self.n_resources;
        let dim = n_agents * n_res;

        // Start with proportional allocation
        let total_budget: f64 = self.agents.iter().map(|a| a.budget).sum();
        let mut x = DVector::zeros(dim);
        for i in 0..n_agents {
            let share = self.agents[i].budget / total_budget;
            for j in 0..n_res {
                x[i * n_res + j] = share * self.total_resources[j] / (1.0 + self.agents[i].risk_aversion);
            }
        }

        // Project onto feasible set
        let obj = SocialWelfareObjective {
            agents: self.agents.clone(),
            n_resources: n_res,
        };

        // Use projected gradient descent
        let gd = GradientDescent::new(
            StepSize::Backtracking { alpha: 1.0, beta: 0.8, c: 1e-4 },
            2000,
            1e-8,
        );

        let result = gd.optimize(&obj, &x);

        // Project onto feasible set
        let mut allocs = Vec::new();
        let mut utilities = Vec::new();
        let mut feasible = true;

        for i in 0..n_agents {
            let mut alloc = result.x.rows(i * n_res, n_res).into_owned();
            // Project onto non-negative orthant and budget
            for j in 0..n_res {
                alloc[j] = alloc[j].max(0.0);
            }
            let agent_total: f64 = alloc.sum();
            if agent_total > self.agents[i].budget {
                alloc *= self.agents[i].budget / agent_total;
            }
            utilities.push(self.agents[i].utility(&alloc));
            allocs.push(alloc);
        }

        // Check resource feasibility
        let mut total_used = DVector::zeros(n_res);
        for alloc in &allocs {
            total_used += alloc;
        }
        for j in 0..n_res {
            if total_used[j] > self.total_resources[j] + 1e-6 {
                feasible = false;
                // Scale down proportionally
                let scale = self.total_resources[j] / total_used[j];
                for alloc in &mut allocs {
                    alloc[j] *= scale;
                }
            }
        }

        let social_welfare: f64 = utilities.iter().sum();

        AllocationResult {
            allocations: allocs,
            social_welfare,
            utilities,
            feasible,
            converged: result.converged,
        }
    }

    /// Proportional fair allocation (each agent gets share proportional to budget).
    pub fn proportional_fair(&self) -> AllocationResult {
        let total_budget: f64 = self.agents.iter().map(|a| a.budget).sum();
        let mut allocs = Vec::new();
        let mut utilities = Vec::new();

        for agent in &self.agents {
            let share = agent.budget / total_budget;
            let alloc: DVector<f64> = self.total_resources.map(|r| r * share);
            utilities.push(agent.utility(&alloc));
            allocs.push(alloc);
        }

        let social_welfare: f64 = utilities.iter().sum();

        AllocationResult {
            allocations: allocs,
            social_welfare,
            utilities,
            feasible: true,
            converged: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_agent_utility() {
        let agent = Agent::new(0, DVector::from_vec(vec![3.0, 2.0]), 0.5, 10.0);
        let x = DVector::from_vec(vec![2.0, 3.0]);
        let util = agent.utility(&x);
        // 3*2 + 2*3 - 0.5*0.5*(4+9) = 12 - 3.25 = 8.75
        assert_abs_diff_eq!(util, 8.75, epsilon = 1e-10);
    }

    #[test]
    fn test_agent_utility_gradient() {
        let agent = Agent::new(0, DVector::from_vec(vec![3.0, 2.0]), 1.0, 10.0);
        let x = DVector::from_vec(vec![1.0, 1.0]);
        let grad = agent.utility_gradient(&x);
        assert_abs_diff_eq!(grad[0], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(grad[1], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_allocation_two_agents() {
        let a1 = Agent::new(0, DVector::from_vec(vec![5.0, 3.0]), 0.1, 10.0);
        let a2 = Agent::new(1, DVector::from_vec(vec![2.0, 4.0]), 0.1, 10.0);
        let total = DVector::from_vec(vec![20.0, 20.0]);
        let problem = ResourceAllocation::new(vec![a1, a2], total);
        let result = problem.allocate();
        assert!(result.feasible);
        assert_eq!(result.allocations.len(), 2);
        // Total allocation shouldn't exceed total resources
        let total_alloc: DVector<f64> = result.allocations.iter().fold(
            DVector::zeros(2),
            |acc, a| acc + a,
        );
        assert!(total_alloc[0] <= 20.0 + 1e-6);
        assert!(total_alloc[1] <= 20.0 + 1e-6);
    }

    #[test]
    fn test_proportional_fair() {
        let a1 = Agent::new(0, DVector::from_vec(vec![5.0]), 0.1, 10.0);
        let a2 = Agent::new(1, DVector::from_vec(vec![3.0]), 0.1, 20.0);
        let total = DVector::from_vec(vec![30.0]);
        let problem = ResourceAllocation::new(vec![a1, a2], total);
        let result = problem.proportional_fair();
        // a1 gets 10/30 * 30 = 10, a2 gets 20/30 * 30 = 20
        assert_abs_diff_eq!(result.allocations[0][0], 10.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result.allocations[1][0], 20.0, epsilon = 1e-10);
    }

    #[test]
    fn test_allocation_nonnegative() {
        let a1 = Agent::new(0, DVector::from_vec(vec![5.0, 3.0]), 0.5, 10.0);
        let a2 = Agent::new(1, DVector::from_vec(vec![2.0, 4.0]), 0.5, 10.0);
        let total = DVector::from_vec(vec![10.0, 10.0]);
        let problem = ResourceAllocation::new(vec![a1, a2], total);
        let result = problem.allocate();
        for alloc in &result.allocations {
            for &v in alloc.iter() {
                assert!(v >= -1e-6, "Negative allocation: {}", v);
            }
        }
    }

    #[test]
    fn test_social_welfare_positive() {
        let a1 = Agent::new(0, DVector::from_vec(vec![5.0]), 0.1, 10.0);
        let a2 = Agent::new(1, DVector::from_vec(vec![3.0]), 0.1, 10.0);
        let total = DVector::from_vec(vec![20.0]);
        let problem = ResourceAllocation::new(vec![a1, a2], total);
        let result = problem.allocate();
        assert!(result.social_welfare > 0.0);
    }

    #[test]
    fn test_allocation_respects_budget() {
        let a1 = Agent::new(0, DVector::from_vec(vec![10.0]), 0.1, 5.0);
        let a2 = Agent::new(1, DVector::from_vec(vec![10.0]), 0.1, 5.0);
        let total = DVector::from_vec(vec![100.0]);
        let problem = ResourceAllocation::new(vec![a1, a2], total);
        let result = problem.allocate();
        // Budget of each agent is 5
        for alloc in &result.allocations {
            assert!(alloc.sum() <= 5.0 + 1e-6);
        }
    }

    #[test]
    fn test_multi_resource_allocation() {
        let a1 = Agent::new(0, DVector::from_vec(vec![3.0, 5.0, 2.0]), 0.2, 15.0);
        let a2 = Agent::new(1, DVector::from_vec(vec![4.0, 2.0, 6.0]), 0.2, 15.0);
        let a3 = Agent::new(2, DVector::from_vec(vec![1.0, 3.0, 4.0]), 0.2, 10.0);
        let total = DVector::from_vec(vec![30.0, 30.0, 30.0]);
        let problem = ResourceAllocation::new(vec![a1, a2, a3], total);
        let result = problem.allocate();
        assert_eq!(result.allocations.len(), 3);
        assert!(result.social_welfare > 0.0);
    }
}
