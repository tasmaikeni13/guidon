import Guidon.Core

/-! Signal-floor and deterministic coefficient-error bounds. No probability model
or Transformer smoothness is assumed or concluded by this file. -/
namespace Guidon
noncomputable section
variable {ι : Type*} [Fintype ι]

/-- The numerical second projection is an exact-real identity. -/
theorem projection_idempotent (a c : ι → ℝ) :
    project a (project a c) = project a c := by
  funext i
  simp [project, projection_neutral]

/-- Deterministic transfer of a coordinatewise correction budget. -/
theorem dot_error_l1 (e δ : ι → ℝ) (ρ : ℝ)
    (hδ : ∀ i, |δ i| ≤ ρ) :
    |dot e δ| ≤ ρ * ∑ i, |e i| := by
  calc
    |dot e δ| ≤ ∑ i, |e i * δ i| := by
      exact Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ i, ρ * |e i| := by
      apply Finset.sum_le_sum
      intro i _
      rw [abs_mul, mul_comm ρ]
      exact mul_le_mul_of_nonneg_left (hδ i) (abs_nonneg _)
    _ = ρ * ∑ i, |e i| := by rw [Finset.mul_sum]

/-- Current coefficient gain can lose at most the stated drift budget. -/
theorem stale_gain_lower (current stored δ : ι → ℝ) (ρ : ℝ)
    (hδ : ∀ i, |δ i| ≤ ρ) :
    dot stored δ - ρ * ∑ i, |current i - stored i| ≤ dot current δ := by
  have hb := dot_error_l1 (fun i => current i - stored i) δ ρ hδ
  have he : dot (fun i => current i - stored i) δ =
      dot current δ - dot stored δ := by
    rw [dot_comm, dot_sub_right, dot_comm δ current, dot_comm δ stored]
  rw [he] at hb
  have hl := (abs_le.mp hb).1
  linarith

/-- Sample neutrality only bounds population error when coefficient error is small. -/
theorem population_neutrality_error (population sample δ : ι → ℝ) (ρ : ℝ)
    (hδ : ∀ i, |δ i| ≤ ρ) (hneut : dot sample δ = 0) :
    |dot population δ| ≤ ρ * ∑ i, |population i - sample i| := by
  have hb := dot_error_l1 (fun i => population i - sample i) δ ρ hδ
  have he : dot (fun i => population i - sample i) δ = dot population δ := by
    rw [dot_comm, dot_sub_right, dot_comm δ population, dot_comm δ sample, hneut]
    ring
  rwa [he] at hb

variable [Nonempty ι]

def floorScale (q : ι → ℝ) (ρ κ : ℝ) : ℝ := ρ / max (maxAbs q) κ

theorem floorScale_nonneg (q : ι → ℝ) (ρ κ : ℝ) (hρ : 0 ≤ ρ) (hκ : 0 < κ) :
    0 ≤ floorScale q ρ κ := by
  exact div_nonneg hρ (le_max_right (maxAbs q) κ |>.trans' hκ.le)

/-- A denominator floor in the SCALE preserves the radius; projection is unchanged. -/
theorem floor_correction_bounded (q : ι → ℝ) (ρ κ : ℝ)
    (hρ : 0 ≤ ρ) (hκ : 0 < κ) (i : ι) : |floorScale q ρ κ * q i| ≤ ρ := by
  have hd : 0 < max (maxAbs q) κ := hκ.trans_le (le_max_right _ _)
  unfold floorScale
  rw [abs_mul, abs_of_nonneg (div_nonneg hρ hd.le)]
  calc
    ρ / max (maxAbs q) κ * |q i| ≤
        ρ / max (maxAbs q) κ * max (maxAbs q) κ :=
      mul_le_mul_of_nonneg_left ((abs_le_maxAbs q i).trans (le_max_left _ _))
        (div_nonneg hρ hd.le)
    _ = ρ := div_mul_cancel₀ ρ (ne_of_gt hd)

theorem floor_weight_interval (q : ι → ℝ) (ρ κ : ℝ)
    (hρ : 0 ≤ ρ) (hκ : 0 < κ) (i : ι) :
    1 - ρ ≤ weights q (floorScale q ρ κ) i ∧
      weights q (floorScale q ρ κ) i ≤ 1 + ρ := by
  have h := abs_le.mp (floor_correction_bounded q ρ κ hρ hκ i)
  unfold weights
  constructor <;> linarith [h.1, h.2]

theorem floor_weights_positive (q : ι → ℝ) (ρ κ : ℝ)
    (hρ : 0 ≤ ρ) (hκ : 0 < κ) (hρ1 : ρ < 1) (i : ι) :
    0 < weights q (floorScale q ρ κ) i := by
  have h := (floor_weight_interval q ρ κ hρ hκ i).1
  linarith

theorem floor_perturbation_energy (q e : ι → ℝ) (ρ κ : ℝ)
    (hρ : 0 ≤ ρ) (hκ : 0 < κ) (he : ∀ i, 0 ≤ e i) :
    (∑ i, (floorScale q ρ κ * q i)^2 * e i) ≤ ρ^2 * ∑ i, e i := by
  rw [Finset.mul_sum]
  apply Finset.sum_le_sum
  intro i _
  have hb := floor_correction_bounded q ρ κ hρ hκ i
  have hs : (floorScale q ρ κ * q i)^2 ≤ ρ^2 := by
    have ha := abs_nonneg (floorScale q ρ κ * q i)
    have ht := sq_abs (floorScale q ρ κ * q i)
    nlinarith
  exact mul_le_mul_of_nonneg_right hs (he i)
end
end Guidon
