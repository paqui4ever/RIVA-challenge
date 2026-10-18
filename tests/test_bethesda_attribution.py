import os
import sys
import numpy as np
import pandas as pd

# Add root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from interpretability_analysis.bethesda_concept_attribution import (
    find_annotations_files,
    load_all_annotations,
    find_image_directory,
    compute_bethesda_enrichment_metrics,
    run_bsf_subspace_probe,
    plot_bethesda_heatmap,
    plot_bsf_dysplasia_manifold_projection
)

def test_pipeline_components():
    print("1. Testing annotation auto-discovery...")
    ann_files = find_annotations_files()
    assert len(ann_files) > 0, "No annotation files found!"
    print(f"Found {len(ann_files)} files: {ann_files}")

    print("2. Testing annotation loading...")
    ann_dict, df = load_all_annotations(ann_files)
    assert len(ann_dict) > 0, "Annotation dict is empty!"
    assert len(df) > 0, "DataFrame is empty!"
    print(f"Loaded {len(df)} cells across {len(ann_dict)} images.")

    print("3. Testing image directory discovery...")
    img_dir = find_image_directory()
    assert os.path.isdir(img_dir), f"Image dir {img_dir} does not exist!"
    print(f"Found image dir: {img_dir}")

    print("4. Testing quantitative metric calculations on synthetic activations...")
    num_concepts = 32
    mock_records = []
    classes = ["NILM", "LSIL", "HSIL", "SCC", "ASC-US", "ASC-H", "ENDOCERVICAL", "INFL"]

    np.random.seed(42)
    for i in range(240):
        cls = np.random.choice(classes)
        act = np.random.exponential(scale=0.08, size=num_concepts)
        if cls == "HSIL":
            act[5] += 2.5
        if cls == "NILM":
            act[10] += 2.0

        subspace = np.random.randn(num_concepts, 4)
        mock_records.append({
            "image_path": "dummy.png",
            "image_filename": "dummy.png",
            "class_name": cls,
            "is_cell": True,
            "cx_pad": 500.0, "cy_pad": 500.0,
            "x1_pad": 450.0, "y1_pad": 450.0, "x2_pad": 550.0, "y2_pad": 550.0,
            "r_grid": 31, "c_grid": 31,
            "pooled_activation": act,
            "subspace_coords": subspace
        })

    # Add BG_EXUDATE tokens
    for i in range(60):
        act = np.random.exponential(scale=0.015, size=num_concepts)
        subspace = np.random.randn(num_concepts, 4)
        mock_records.append({
            "image_path": "dummy.png",
            "image_filename": "dummy.png",
            "class_name": "BG_EXUDATE",
            "is_cell": False,
            "cx_pad": 200.0, "cy_pad": 200.0,
            "x1_pad": 184.0, "y1_pad": 184.0, "x2_pad": 216.0, "y2_pad": 216.0,
            "r_grid": 12, "c_grid": 12,
            "pooled_activation": act,
            "subspace_coords": subspace
        })

    test_out = "./interpretability_analysis/test_attribution"
    os.makedirs(test_out, exist_ok=True)

    metrics_csv = os.path.join(test_out, "test_metrics.csv")
    df_metrics = compute_bethesda_enrichment_metrics(mock_records, num_concepts, metrics_csv)
    assert os.path.exists(metrics_csv), "Metrics CSV was not written!"
    assert df_metrics.loc[5, "top_bethesda_class"] == "HSIL", f"Expected C5 top class HSIL, got {df_metrics.loc[5, 'top_bethesda_class']}"
    assert df_metrics.loc[10, "top_bethesda_class"] == "NILM", f"Expected C10 top class NILM, got {df_metrics.loc[10, 'top_bethesda_class']}"
    assert df_metrics.loc[5, "cell_vs_bg_ratio"] > 1.0
    print(f"Concept 5 selectivity for HSIL: {df_metrics.loc[5, 'selectivity_HSIL']:.2f}")
    print(f"Concept 10 selectivity for NILM: {df_metrics.loc[10, 'selectivity_NILM']:.2f}")

    print("5. Testing BSF subspace linear probe...")
    probe_csv = os.path.join(test_out, "test_probe.csv")
    df_probe = run_bsf_subspace_probe(mock_records, df_metrics, probe_csv, top_n_blocks=5)
    assert os.path.exists(probe_csv), "Probe CSV was not written!"
    assert len(df_probe) > 0, "Probe DataFrame empty!"
    print(f"BSF Probe evaluated on {len(df_probe)} blocks. Sample acc: {df_probe['cv_accuracy_mean'].iloc[0]:.3f}")

    print("6. Testing clustered heatmap generation...")
    heatmap_png = os.path.join(test_out, "test_heatmap.png")
    plot_bethesda_heatmap(df_metrics, "TestModel", heatmap_png, top_n_concepts=10)
    assert os.path.exists(heatmap_png), "Heatmap PNG was not created!"
    print(f"Heatmap saved ({os.path.getsize(heatmap_png)} bytes).")

    print("7. Testing BSF dysplasia manifold projection...")
    manifold_png = os.path.join(test_out, "test_manifold.png")
    plot_bsf_dysplasia_manifold_projection(mock_records, df_metrics, manifold_png, target_blocks=[5, 10])
    assert os.path.exists(manifold_png), "Manifold PNG was not created!"
    print(f"Manifold projection saved ({os.path.getsize(manifold_png)} bytes).")

    print("\n ALL COMPONENT TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_pipeline_components()
