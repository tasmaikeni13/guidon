import Guidon.Core

namespace Guidon

/-- A public specification of allowed training input. Evaluation is a separate
payload and is not an argument to any state transition. -/
def trajectory {State Input : Type*} (step : State → Input → State)
    (initial : State) (inputs : Nat → Input) : Nat → State
  | 0 => initial
  | n + 1 => step (trajectory step initial inputs n) (inputs n)

def isolatedRun {State Input Evaluation : Type*} (step : State → Input → State)
    (initial : State) (inputs : Nat → Input) (_evaluation : Evaluation) (n : Nat) :=
  trajectory step initial inputs n

/-- P9: any two evaluation payloads yield the same model trajectory, provided
all allowed inputs and initialization really are independent of evaluation. -/
theorem evaluation_noninterference {State Input Evaluation : Type*}
    (step : State → Input → State) (initial : State) (inputs : Nat → Input)
    (eval₁ eval₂ : Evaluation) (n : Nat) :
    isolatedRun step initial inputs eval₁ n = isolatedRun step initial inputs eval₂ n := rfl

theorem transcript_extensionality {State Input : Type*}
    (step : State → Input → State) (initial : State) (x y : Nat → Input)
    (hxy : ∀ n, x n = y n) (n : Nat) :
    trajectory step initial x n = trajectory step initial y n := by
  induction n with
  | zero => rfl
  | succ n ih => simp only [trajectory, ih, hxy]

/-- P10: reducing a training/guidance objective can increase independent
evaluation loss, so no universal generalization claim follows from P1--P9. -/
theorem generalization_counterexample :
    ((1 : ℝ) - 1)^2 < ((0 : ℝ) - 1)^2 ∧
    ((0 : ℝ) + 1)^2 < ((1 : ℝ) + 1)^2 := by norm_num

end Guidon
