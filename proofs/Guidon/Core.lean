import Mathlib

/-! Exact-real coefficient algebra for GUIDON. This file does not model
floating-point kernels, neural-network gradients, or distributional assumptions. -/
namespace Guidon

noncomputable section

variable {ι : Type*} [Fintype ι]

def dot (x y : ι → ℝ) : ℝ := ∑ i, x i * y i

def project (a c : ι → ℝ) : ι → ℝ :=
  fun i => c i - (dot a c / dot a a) * a i

def weights (q : ι → ℝ) (τ : ℝ) : ι → ℝ := fun i => 1 + τ * q i

lemma dot_comm (x y : ι → ℝ) : dot x y = dot y x := by
  unfold dot
  congr 1
  ext i
  ring

lemma dot_add_right (x y z : ι → ℝ) :
    dot x (fun i => y i + z i) = dot x y + dot x z := by
  simp [dot, mul_add, Finset.sum_add_distrib]

lemma dot_sub_right (x y z : ι → ℝ) :
    dot x (fun i => y i - z i) = dot x y - dot x z := by
  simp [dot, mul_sub, Finset.sum_sub_distrib]

lemma dot_scale_right (x y : ι → ℝ) (r : ℝ) :
    dot x (fun i => r * y i) = r * dot x y := by
  unfold dot
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro i _
  ring

lemma dot_self_nonneg (x : ι → ℝ) : 0 ≤ dot x x := by
  exact Finset.sum_nonneg (fun i _ => mul_self_nonneg (x i))

lemma dot_self_zero (a : ι → ℝ) (ha : dot a a = 0) : ∀ i, a i = 0 := by
  have hz : ∀ i ∈ (Finset.univ : Finset ι), a i * a i = 0 :=
    (Finset.sum_eq_zero_iff_of_nonneg (fun i _ => mul_self_nonneg (a i))).mp ha
  intro i
  exact (mul_self_eq_zero).mp (hz i (Finset.mem_univ i))

/-- P1: the coefficient correction is orthogonal to training progress. -/
theorem projection_neutral (a c : ι → ℝ) : dot a (project a c) = 0 := by
  unfold project
  rw [dot_sub_right, dot_scale_right]
  by_cases ha : dot a a = 0
  · have hz := dot_self_zero a ha
    simp [dot, hz]
  · field_simp
    ring

/-- P2: projection captures exactly its squared norm in guide alignment. -/
theorem projection_gain (a c : ι → ℝ) :
    dot c (project a c) = dot (project a c) (project a c) := by
  have h := projection_neutral a c
  have he : c = fun i => project a c i + (dot a c / dot a a) * a i := by
    funext i
    simp [project]
  calc
    dot c (project a c) = dot (project a c) c := dot_comm _ _
    _ = dot (project a c)
        (fun i => project a c i + (dot a c / dot a a) * a i) :=
      congrArg (dot (project a c)) he
    _ = dot (project a c) (project a c) +
        (dot a c / dot a a) * dot (project a c) a := by
      rw [dot_add_right, dot_scale_right]
    _ = dot (project a c) (project a c) := by
      have h' : dot (project a c) a = 0 := by rw [dot_comm]; exact h
      rw [h']
      ring

/-- P3: the reweighted adaptive direction preserves training progress. -/
theorem training_progress (a c : ι → ℝ) (τ : ℝ) :
    dot a (weights (project a c) τ) = dot a (fun _ => 1) := by
  unfold weights
  rw [dot_add_right, dot_scale_right, projection_neutral]
  ring

/-- P4: the fresh guide linear model improves by τ times a squared norm. -/
theorem guide_progress (a c : ι → ℝ) (τ : ℝ) :
    dot c (weights (project a c) τ) =
      dot c (fun _ => 1) + τ * dot (project a c) (project a c) := by
  unfold weights
  rw [dot_add_right, dot_scale_right, projection_gain]

theorem guide_nonworsening (a c : ι → ℝ) (τ : ℝ) (hτ : 0 ≤ τ) :
    dot c (fun _ => 1) ≤ dot c (weights (project a c) τ) := by
  rw [guide_progress]
  exact le_add_of_nonneg_right (mul_nonneg hτ (dot_self_nonneg _))

theorem zero_radius (a c : ι → ℝ) : weights (project a c) 0 = fun _ => 1 := by
  funext i
  simp [weights]

lemma projection_on_feasible (a c z : ι → ℝ) (hz : dot a z = 0) :
    dot c z = dot (project a c) z := by
  have he : c = fun i => project a c i + (dot a c / dot a a) * a i := by
    funext i
    simp [project]
  calc
    dot c z = dot z c := dot_comm _ _
    _ = dot z (fun i => project a c i + (dot a c / dot a a) * a i) :=
      congrArg (dot z) he
    _ = dot z (project a c) + (dot a c / dot a a) * dot z a := by
      rw [dot_add_right, dot_scale_right]
    _ = dot (project a c) z := by
      have hza : dot z a = 0 := by rw [dot_comm]; exact hz
      rw [hza, dot_comm z (project a c)]
      ring

/-- P11: the projected direction optimizes a quadratic-regularized linear
guide objective over all training-neutral coefficient corrections. -/
theorem quadratic_optimality (a c z : ι → ℝ) (τ : ℝ) (hz : dot a z = 0) :
    2 * τ * dot c z - dot z z ≤ τ^2 * dot (project a c) (project a c) := by
  let q := project a c
  have hn : 0 ≤ ∑ i, (z i - τ * q i)^2 :=
    Finset.sum_nonneg (fun i _ => sq_nonneg _)
  have he : (∑ i, (z i - τ * q i)^2) =
      dot z z - 2 * τ * dot q z + τ^2 * dot q q := by
    calc
      (∑ i, (z i - τ * q i)^2) =
          ∑ i, (z i * z i - 2 * τ * (q i * z i) + τ^2 * (q i * q i)) := by
        apply Finset.sum_congr rfl
        intro i _
        ring
      _ = _ := by
        simp [dot, Finset.sum_sub_distrib, Finset.sum_add_distrib, Finset.mul_sum]
  rw [projection_on_feasible a c z hz]
  change 2 * τ * dot q z - dot z z ≤ τ^2 * dot q q
  linarith

theorem quadratic_optimum_attained (a c : ι → ℝ) (τ : ℝ) :
    2 * τ * dot c (fun i => τ * project a c i) -
      dot (fun i => τ * project a c i) (fun i => τ * project a c i) =
      τ^2 * dot (project a c) (project a c) := by
  rw [dot_scale_right, dot_scale_right, dot_comm (fun i => τ * project a c i),
    dot_scale_right, projection_gain]
  ring

variable [Nonempty ι]

def maxAbs (q : ι → ℝ) : ℝ :=
  Finset.univ.sup' Finset.univ_nonempty (fun i => |q i|)

def stepScale (q : ι → ℝ) (ρ : ℝ) : ℝ :=
  if maxAbs q = 0 then 0 else ρ / maxAbs q

lemma abs_le_maxAbs (q : ι → ℝ) (i : ι) : |q i| ≤ maxAbs q := by
  exact Finset.le_sup' (fun j => |q j|) (Finset.mem_univ i)

lemma maxAbs_nonneg (q : ι → ℝ) : 0 ≤ maxAbs q := by
  obtain ⟨i⟩ := ‹Nonempty ι›
  exact (abs_nonneg (q i)).trans (abs_le_maxAbs q i)

def normalized (x : ι → ℝ) : ι → ℝ :=
  if maxAbs x = 0 then fun _ => 0 else fun i => x i / maxAbs x

theorem normalized_reconstruction (x : ι → ℝ) (i : ι) :
    x i = maxAbs x * normalized x i := by
  by_cases hx : maxAbs x = 0
  · have hi := abs_le_maxAbs x i
    rw [hx] at hi
    have hz : x i = 0 := abs_eq_zero.mp (le_antisymm hi (abs_nonneg _))
    simp [normalized, hx, hz]
  · simp only [normalized, hx, ↓reduceIte]
    field_simp

lemma dot_normalized_reconstruction (x z : ι → ℝ) :
    dot x z = maxAbs x * dot (normalized x) z := by
  have he : x = fun i => maxAbs x * normalized x i :=
    funext (normalized_reconstruction x)
  calc
    dot x z = dot z x := dot_comm _ _
    _ = dot z (fun i => maxAbs x * normalized x i) := congrArg (dot z) he
    _ = maxAbs x * dot (normalized x) z := by
      rw [dot_scale_right, dot_comm z (normalized x)]

/-- P3 bridge: the normalized implementation preserves RAW training progress. -/
theorem normalized_training_progress (a c : ι → ℝ) (τ : ℝ) :
    dot a (weights (project (normalized a) (normalized c)) τ) =
      dot a (fun _ => 1) := by
  rw [dot_normalized_reconstruction a, training_progress,
    ← dot_normalized_reconstruction a]

/-- P4 bridge: the fresh RAW guide gain includes the positive normalization factor. -/
theorem normalized_guide_progress (a c : ι → ℝ) (τ : ℝ) :
    dot c (weights (project (normalized a) (normalized c)) τ) =
      dot c (fun _ => 1) + maxAbs c * τ *
        dot (project (normalized a) (normalized c))
          (project (normalized a) (normalized c)) := by
  rw [dot_normalized_reconstruction c, guide_progress,
    dot_normalized_reconstruction c (fun _ => 1)]
  ring

theorem stepScale_nonneg (q : ι → ℝ) (ρ : ℝ) (hρ : 0 ≤ ρ) :
    0 ≤ stepScale q ρ := by
  unfold stepScale
  split_ifs
  · exact le_rfl
  · exact div_nonneg hρ (maxAbs_nonneg q)

/-- P5: the actual infinity-norm normalization satisfies the radius bound. -/
theorem correction_bounded (q : ι → ℝ) (ρ : ℝ) (hρ : 0 ≤ ρ) (i : ι) :
    |stepScale q ρ * q i| ≤ ρ := by
  unfold stepScale
  split_ifs with hz
  · simpa using hρ
  · have hm : 0 < maxAbs q := lt_of_le_of_ne (maxAbs_nonneg q) (Ne.symm hz)
    rw [abs_mul, abs_of_nonneg (div_nonneg hρ hm.le)]
    calc
      ρ / maxAbs q * |q i| ≤ ρ / maxAbs q * maxAbs q :=
        mul_le_mul_of_nonneg_left (abs_le_maxAbs q i) (div_nonneg hρ hm.le)
      _ = ρ := div_mul_cancel₀ ρ hz

theorem weight_interval (q : ι → ℝ) (ρ : ℝ) (hρ : 0 ≤ ρ) (i : ι) :
    1 - ρ ≤ weights q (stepScale q ρ) i ∧
      weights q (stepScale q ρ) i ≤ 1 + ρ := by
  have h := (abs_le.mp (correction_bounded q ρ hρ i))
  unfold weights
  constructor <;> linarith [h.1, h.2]

theorem weights_positive (q : ι → ℝ) (ρ : ℝ) (hρ : 0 ≤ ρ) (hρ1 : ρ < 1)
    (i : ι) : 0 < weights q (stepScale q ρ) i := by
  have h := (weight_interval q ρ hρ i).1
  linarith

/-- P6: squared parameter perturbation bound for disjoint blocks, where e_i
is the squared Euclidean norm of adaptive block i. -/
theorem perturbation_energy (q e : ι → ℝ) (ρ : ℝ) (hρ : 0 ≤ ρ)
    (he : ∀ i, 0 ≤ e i) :
    (∑ i, (stepScale q ρ * q i)^2 * e i) ≤ ρ^2 * ∑ i, e i := by
  rw [Finset.mul_sum]
  apply Finset.sum_le_sum
  intro i _
  have hb := correction_bounded q ρ hρ i
  have hs : (stepScale q ρ * q i)^2 ≤ ρ^2 := by
    have ha := abs_nonneg (stepScale q ρ * q i)
    have ht := sq_abs (stepScale q ρ * q i)
    nlinarith
  exact mul_le_mul_of_nonneg_right hs (he i)

/-- P7: Taylor-remainder transfer. Smoothness and positive baseline progress
are assumptions; this is not an unconditional AdamW convergence theorem. -/
theorem conditional_descent (before after η progress curvature normSq : ℝ)
    (hη : 0 ≤ η) (hb : after ≤ before - η * progress + curvature * η^2 * normSq / 2)
    (hbudget : curvature * η * normSq ≤ 2 * progress) : after ≤ before := by
  have h := mul_le_mul_of_nonneg_left hbudget hη
  nlinarith

/-- P8: comparison of actual guide losses under explicitly supplied
upper/lower Taylor remainder bounds for the two different steps. -/
theorem guide_loss_comparison (before base guided η progress gain rBase rGuided : ℝ)
    (hupper : guided ≤ before - η * (progress + gain) + rGuided)
    (hlower : before - η * progress - rBase ≤ base)
    (hdominates : rBase + rGuided ≤ η * gain) : guided ≤ base := by
  nlinarith

end

end Guidon
