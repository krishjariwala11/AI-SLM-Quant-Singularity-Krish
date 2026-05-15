import json
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent
NO_RAG_PATH = DATA_DIR / "data" / "eval_results_no_rag.json"
RAG_PATH = DATA_DIR / "data" / "eval_results_rag.json"
REPORT_PATH = DATA_DIR / "report" / "report.md"

def generate_results_markdown(no_rag_results, rag_results):
    md = "## Section 4: Results\n\n"
    
    # 1. Walk-forward directional accuracy
    md += "### Walk-Forward Directional Accuracy (Per 5-Day Window)\n\n"
    md += "| Block | Days | Accuracy (No RAG) | Accuracy (With RAG) |\n"
    md += "|-------|------|-------------------|--------------------|\n"
    
    no_rag_blocks = no_rag_results.get("per_block_accuracy", {})
    rag_blocks = rag_results.get("per_block_accuracy", {}) if rag_results else {}
    
    for block_name, data in no_rag_blocks.items():
        rag_data = rag_blocks.get(block_name, {})
        rag_acc = f"{rag_data.get('accuracy', 0):.3f}" if rag_data else "N/A"
        md += f"| {block_name} | {data['count']} | {data['accuracy']:.3f} | {rag_acc} |\n"
        
    ci_no = no_rag_results.get("confidence_intervals", {}).get("directional_accuracy", {})
    md += f"\n**Overall Accuracy (No RAG)**: {no_rag_results.get('overall_directional_accuracy', 0):.3f} "
    md += f"(95% CI: [{ci_no.get('ci_95_lower', 0):.3f}, {ci_no.get('ci_95_upper', 0):.3f}])\n"
    
    if rag_results:
        ci_rag = rag_results.get("confidence_intervals", {}).get("directional_accuracy", {})
        md += f"**Overall Accuracy (With RAG)**: {rag_results.get('overall_directional_accuracy', 0):.3f} "
        md += f"(95% CI: [{ci_rag.get('ci_95_lower', 0):.3f}, {ci_rag.get('ci_95_upper', 0):.3f}])\n"

    # 2. Output Schema Pass Rate
    md += "\n### Output Schema Pass Rate\n\n"
    md += f"- **No RAG**: {no_rag_results.get('schema_pass_rate', 0):.1%}\n"
    if rag_results:
        md += f"- **With RAG**: {rag_results.get('schema_pass_rate', 0):.1%}\n"

    # 3. Conviction Reliability Across Bins
    md += "\n### Conviction Reliability Across Bins (No RAG)\n\n"
    md += "| Conviction Bin | Count | Directional Accuracy |\n"
    md += "|----------------|-------|----------------------|\n"
    
    bins = no_rag_results.get("conviction_calibration", {}).get("bins", {})
    for bin_label, bin_data in bins.items():
        acc = f"{bin_data.get('accuracy', 0):.3f}" if bin_data.get('accuracy') is not None else "N/A"
        md += f"| {bin_label} | {bin_data.get('count', 0)} | {acc} |\n"
        
    is_mono = no_rag_results.get("conviction_calibration", {}).get("is_monotonic", False)
    md += f"\n**Calibration is monotonic**: {'Yes' if is_mono else 'No'}\n"

    # 4. Orchestrator Rates & VIX Regime
    md += "\n### Orchestrator Metrics & Regime Performance\n\n"
    md += "| Metric | No RAG | With RAG | Status (Threshold) |\n"
    md += "|--------|--------|----------|--------------------|\n"
    
    o_no = no_rag_results.get("orchestrator_rates", {})
    o_rag = rag_results.get("orchestrator_rates", {}) if rag_results else {}
    assess = no_rag_results.get("assessment", {})
    
    supp_status = assess.get("suppression_rate", {}).get("status", "N/A")
    down_status = assess.get("downgrade_rate", {}).get("status", "N/A")
    parse_status = assess.get("parse_failure_rate", {}).get("status", "N/A")
    vix_status = assess.get("vix_regime_gap", {}).get("status", "N/A")
    
    md += f"| Suppression Rate | {o_no.get('suppression_rate', 0):.1%} | {o_rag.get('suppression_rate', 0) if o_rag else 'N/A'} | {supp_status} |\n"
    md += f"| Downgrade Rate | {o_no.get('downgrade_rate', 0):.1%} | {o_rag.get('downgrade_rate', 0) if o_rag else 'N/A'} | {down_status} |\n"
    md += f"| Parse Failure Rate | {o_no.get('parse_failure_rate', 0):.1%} | {o_rag.get('parse_failure_rate', 0) if o_rag else 'N/A'} | {parse_status} |\n"
    
    v_no = no_rag_results.get("vix_regime", {})
    v_rag = rag_results.get("vix_regime", {}) if rag_results else {}
    md += f"| High VIX Accuracy | {v_no.get('high_vix_accuracy', 0):.3f} | {v_rag.get('high_vix_accuracy', 0) if v_rag else 'N/A'} | N/A |\n"
    md += f"| Low VIX Accuracy | {v_no.get('low_vix_accuracy', 0):.3f} | {v_rag.get('low_vix_accuracy', 0) if v_rag else 'N/A'} | N/A |\n"
    md += f"| VIX Regime Gap | {v_no.get('accuracy_gap', 0):.3f} | {v_rag.get('accuracy_gap', 0) if v_rag else 'N/A'} | {vix_status} |\n"
    
    return md

def update_report():
    if not NO_RAG_PATH.exists():
        print(f"Error: {NO_RAG_PATH} not found. Please run 'python src/run_eval.py' first.")
        return
        
    with open(NO_RAG_PATH, "r") as f:
        no_rag_results = json.load(f)
        
    rag_results = None
    if RAG_PATH.exists():
        with open(RAG_PATH, "r") as f:
            rag_results = json.load(f)
            
    if not REPORT_PATH.exists():
        print(f"Error: {REPORT_PATH} not found.")
        return
        
    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        content = f.read()
        
    # Find the Section 4 placeholder and replace it
    start_idx = content.find("## Section 4: Results")
    end_idx = content.find("## Section 5: How Do I Know This Pod Is Safe to Connect?")
    
    if start_idx == -1 or end_idx == -1:
        print("Error: Could not find Section 4 or Section 5 markers in report.md.")
        return
        
    new_section = generate_results_markdown(no_rag_results, rag_results)
    
    new_content = content[:start_idx] + new_section + "\n---\n\n" + content[end_idx:]
    
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(new_content)
        
    print(f"Successfully updated Section 4 in {REPORT_PATH}")

if __name__ == "__main__":
    update_report()
