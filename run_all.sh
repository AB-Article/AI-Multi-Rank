#!/usr/bin/env bash
#
# Reproduce every number, table and figure in the paper, the Online Appendix and
# the technical report, from the data in this archive.
#
#   ./run_all.sh            full pipeline: 160 scripts
#   ./run_all.sh fast       the same, less the eight slowest stages
#   ./run_all.sh guards     the six consistency guards only (about a minute)
#   ./run_all.sh list       print the execution order and exit
#
# Wall clock depends on the machine and is dominated by a handful of stages, so
# no single figure is quoted here.  Each stage prints its own elapsed seconds as
# it finishes and the log under results/logs/ keeps them, so a run measures
# itself.  For scale, on two cores: most stages take seconds, but 07, 09 and 14
# take three to six minutes each, and the eight stages that 'fast' skips are the
# longest of all -- 82, 116, 130 and 167 are tens of minutes apiece.  Budget
# hours, not minutes, for a full run, and start with 'fast'.
#
# Inputs are read from data/ and all outputs are written to results/ and
# figures/.  Nothing outside those two directories is modified, so a run can
# always be checked against the archived copies by diffing them:
#
#   ./run_all.sh && git diff --stat results figures
#
# Override the locations with DATA= and OUT= if you keep them elsewhere.
#
# NO NETWORK ACCESS is needed or attempted.  The three scripts that can reach
# the network are excluded from every mode here: 00_fetch_data.py and
# 13_factorial_grids.py rebuild data/ from upstream (see README, "Full rebuild
# from upstream sources"), and 49_pairwise_floor.py --panel builds an auxiliary
# panel.  None of the three is needed for any published number, and
# 63_cofailure_panel.py runs offline against the cache already in
# data/cofailure/.
#
# Every random seed in this archive is fixed, so a rerun is reproducible on a
# given platform.  The two Monte-Carlo additions, 167 and 168, were verified
# bit-identical across two independent runs.

set -uo pipefail

export DATA="${DATA:-$(pwd)/data}"
export OUT="${OUT:-$(pwd)/results}"
FIGS="${FIGS:-$(pwd)/figures}"
MODE="${1:-full}"

mkdir -p "$OUT/logs" "$FIGS"

FAILED=0
SKIPPED=0

# Stages the provenance map documents as slow, by script number.  'fast' mode
# skips exactly these: 30, 63, 82, 116, 124, 127, 130 and 167.  They are the
# longest stages of a full run and dominate its wall clock.
SLOW=" 30 63 82 116 124 127 130 167 "

run () {
    local script="$1"; shift
    local name="${script%.py}"
    local num="${script%%_*}"
    if [ "$MODE" = "fast" ] && [ "${SLOW#* $num }" != "$SLOW" ]; then
        printf '  %-38s skipped (slow; fast mode)\n' "$name"
        SKIPPED=$((SKIPPED+1)); return 0
    fi
    printf '  %-38s ' "$name"
    local t0=$SECONDS
    if ( cd scripts && python3 "$script" "$@" ) > "$OUT/logs/$name.log" 2>&1; then
        printf 'ok  %4ds\n' "$((SECONDS-t0))"
    else
        printf 'FAILED  (see results/logs/%s.log)\n' "$name"
        FAILED=$((FAILED+1))
    fi
}

STAGE1=(
    01_build_frontiers.py
    02_verify_theory.py
    03_fit_laws.py
    04_extrapolation.py
    05_verify_generalisations.py
    06_selection_example.py
    07_new_corpora.py
    08_cross_field.py
    09_all_fields.py
    10_universal_curve.py
    11_selection_rules.py
    12_figures_rules.py
    14_test_N_equals_n.py
    15_factorial_robustness.py
    16_verify_fixed_ratio.py
    17_factorial_summary.py
    18_figures_factorial.py
    19_minimax_verification.py
    22_holdout_floor.py
    23_resolution_channels.py
    24_floor_free_exponent.py
    25_budget_law.py
    26_shared_shrinkage.py
    27_resolution_model.py
    28_figure_spectral.py
    29_figure_overview.py
    30_surprisal_running_floor.py
    31_figure_surprisal.py
    32_bend_divergence.py
    33_figure_bend.py
    34_path_length.py
)

STAGE2=(
    35_figure_path.py
    36_chord_geodesic.py
    37_figure_chord.py
    38_width_coordinate.py
    39_figure_width.py
    40_width_null_calibration.py
    41_width_window_profile.py
    42_resolution_power.py
    43_floor_axis.py
    44_noise_binomial.py
    45_accuracy_only.py
    46_figure_acconly.py
    51_verify_new_identities.py
    52_floor_cap_bias.py
)

STAGE3=(
    47_exchange_rate.py
    48_exponent_transfer.py
    49_pairwise_floor.py
    50_theory_replacements.py --out ../results/table_theory_replacements.csv
    53_duality.py
    54_am_gm_master.py
    55_axis_selfduality.py
    56_onedim_minimax.py
    57_arc_covering.py
    58_arc_empirical.py
    59_figure_arc.py
    60_sm_panel_summaries.py
    61_three_routes.py
    62_mode_visibility.py
    63_cofailure_panel.py
    64_microfoundation.py
    65_regret_law.py
    66_regret_law_panel.py
    67_figure_regret_micro.py
    68_prediction_assessment.py
    69_spectral_breadth.py
    70_figure_prediction.py
    71_count_axis.py
    72_count_axis_holdout.py
    73_figure_count_axis.py
    74_theory_comparison.py
    75_local_calculus.py
    76_no_floor.py
    77_shape_from_arc.py
    79_figure_nofloor.py
    80_invariant_calculus.py
    81_monotone_floor.py
    82_chart_test.py
    83_exchangeability.py
    84_chart_verdict.py
    86_figure_invariant.py
    87_bernstein_gauge.py
    88_ladder.py
    89_neff.py
    90_knee_threshold.py
    91_knee_archive.py
    92_mode_lower_bound.py
    93_knee_null.py
    95_figure_ladder.py
    96_composition_identities.py
    97_budget_bound.py
    98_divisibility_test.py
    99_cure_chart.py
    100_figure_composition.py
    102_tower_identities.py
    103_price_logconvexity.py
    104_tower_and_arc.py
    105_figure_tower.py
    107_horizon_lattice.py
    108_fourpoint_compute.py
    109_chart_line.py
    110_powerfloor_chart.py
    111_figure_lattice.py
    113_attempt_chain.py
    114_count_hankel.py
    115_admissible_quadruples.py
    116_attempt_chart.py
    117_retry_budget.py
    119_figure_chain.py
    120_two_counts.py
    121_derived_laws.py
    122_elbow_and_wall.py
    123_old_vs_new.py
    124_grid_laws.py
    125_grid_corpus.py
    126_old_vs_new_grid.py
    127_channel_depth_plane.py
    128_power_audit.py
    129_plane_corpus.py
    130_plane_verdict.py
    131_power_table.py
    132_old_vs_new_plane.py
    133_figure_plane.py
    134_ode_first.py
    135_invariant_coords.py
    136_ode_corpus.py
    137_old_vs_new_ode.py
    138_figure_ode.py
    139_class_gate_corpus.py
    140_precision_gate_corpus.py
    141_count_axis_panel.py
    142_undersizing_measured.py
    143_selector_audit.py
    144_multiresource_price.py
    145_paired_band_panel.py
    146_certified_set.py
    147_design_rule.py
    148_sequential_design.py
    149_slope_bias_corrected.py
    150_envelope_threshold.py
    151_certified_ranking.py
    152_selection_grade_price.py
    153_input_size_axis.py
    154_break_even_ladder.py
    155_price_envelope.py
    156_measurement_only.py
    157_input_routes.py
    158_floor_share_null.py
    159_exponent_decomposition.py
    160_arc_deficit_gauge.py
    161_d1hat_spacing.py
    162_verification_index.py
    163_published_rule.py
    164_plugin_regret.py
    165_floor_filter_sensitivity.py
    166_rules_significance.py
    167_ceiling_null.py
    168_rule_spread.py
    169_prediction_clustered.py
    20_nested_verification.py
    21_nested_reference_lm.py
)

if [ "$MODE" = "list" ]; then
    printf '%s\n' "${STAGE1[@]}" "${STAGE2[@]}" "${STAGE3[@]}"
    exit 0
fi

guards () {
    echo "[guards] six consistency checks"
    # A guard reports its offenders on stdout and does not always exit non-zero,
    # so the log is scanned as well as the status: "FOUND n" or "MISMATCH" is a
    # failure however the process exited.
    for g in _guard_titles _guard_prov _guard_nums _guard_readme _guard_refwords _guard_snippet; do
        printf '  %-38s ' "$g"
        ( cd scripts && python3 "$g.py" ) > "$OUT/logs/$g.log" 2>&1
        local status=$?
        if grep -qE 'FOUND [1-9]|MISMATCHES \([1-9]|^  - ' "$OUT/logs/$g.log"; then
            echo "OFFENDERS  (see results/logs/$g.log)"
            FAILED=$((FAILED+1))
        elif [ "$status" -ne 0 ]; then
            echo "FAILED  (see results/logs/$g.log)"
            FAILED=$((FAILED+1))
        elif grep -q 'SKIPPED' "$OUT/logs/$g.log"; then
            echo "skipped (needs the article sources)"
        else
            echo "ok"
        fi
    done
}

if [ "$MODE" = "guards" ]; then guards; exit 0; fi

echo "[1/4] frontiers, laws, factorial grids, spectral extension"
for s in "${STAGE1[@]}"; do run $s; done

echo "[2/4] width coordinate, floor axis, accuracy-only collapse"
for s in "${STAGE2[@]}"; do run $s; done

echo "[3/4] the revisions, in the order they were written"
for s in "${STAGE3[@]}"; do run $s; done

echo "[4/4] guards"
guards

echo
if [ "$FAILED" -eq 0 ]; then
    echo "Done.  No stage failed.  Numbers in results/, figures in figures/."
else
    echo "Done with $FAILED failed stage(s).  See results/logs/."
fi
[ "$SKIPPED" -gt 0 ] && echo "$SKIPPED slow stage(s) skipped; rerun without 'fast' for the full pipeline."

echo "Headline values:"
python3 - <<'PY'
import csv, os
out = os.environ.get("OUT", "results")
def row(name):
    p = os.path.join(out, name)
    if not os.path.exists(p): return None
    with open(p) as f: return next(csv.DictReader(f), None)

r = row("table_ceiling_null_summary.csv")
if r:
    print("  ceiling test, observed violations      %s of %s"
          % (r["incumbent_violations"], r["n_frontiers"]))
    print("  ... expected under the matched null    %.1f   (the count is withdrawn)"
          % float(r["incumbent_expected_under_null"]))
    print("  ... calibrated rejections at 5%%        %s (median triple), %s (widest triple)"
          % (r["med_rejections"], r["wide_rejections"]))
r = row("table_rule_spread_summary.csv")
if r:
    print("  rules in use: distinct picks, median   %.0f of 8 on %s frontiers"
          % (float(r["median_distinct_picks"]), r["n_frontiers"]))
    print("  ... spread between the extreme picks   %.0fx resource for %.2f bits"
          % (float(r["median_spread_cost_mult"]), float(r["median_spread_bits"])))
    print("  ... unit change moves the one rule     %s of %s frontiers"
          % (r["unitflip_moves"], r["n_frontiers"]))
r = row("table_floor_share_null_summary.csv")
if r and "beta" in ",".join(r.keys()):
    print("  zero-floor null: see table_floor_share_null_summary.csv")
PY

exit $(( FAILED > 0 ))
