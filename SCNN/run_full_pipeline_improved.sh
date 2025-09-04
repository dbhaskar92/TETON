#!/bin/bash

# Complete EEG Classification Pipeline - IMPROVED VERSION
# This script runs the entire workflow with the improved export script:
# 1. export_windows_classification_improved.py - processes ALL EEG data with better thresholding
# 2. train_tmx_classification_sccnn.py - trains SCCNN classification model
# 3. Generates comprehensive results and visualizations

echo "=================================================="
echo "Complete EEG Classification Pipeline - IMPROVED VERSION"
echo "=================================================="
echo ""

# Check if we're in the correct directory
if [ ! -f "export_windows_classification_lastcase.py" ] || [ ! -f "train_tmx_classification_sccnn.py" ]; then
    echo "Error: Required Python files not found in current directory"
    echo "Please run this script from /home/nikita/projects/imperial_project/my_code/ppt"
    exit 1
fi

# Check if EEG data directory exists
if [ ! -d "../eeg" ]; then
    echo "Error: EEG data directory not found"
    echo "Please ensure the '../eeg/' directory exists with your EEG data files"
    exit 1
fi

# Check if conda environment is activated
if [[ "$CONDA_DEFAULT_ENV" != "nik_imperial" ]]; then
    echo "Warning: Conda environment 'nik_imperial' is not activated"
    echo "Current environment: $CONDA_DEFAULT_ENV"
    echo "Please activate the environment first: conda activate nik_imperial"
    echo ""
    read -p "Continue anyway? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create output directories
echo "Creating output directories..."
mkdir -p export_tnx_windows_improved
mkdir -p trained_sccnn_model
mkdir -p pipeline_results_improved

echo "✓ Directories created"
echo ""

# Step 1: Process all EEG data and export windows with IMPROVED script
echo "Step 1: Processing EEG Data and Exporting Windows (IMPROVED)"
echo "============================================================="
echo "This step will:"
echo "  • Scan the ../eeg/ directory for all EEG files"
echo "  • Automatically extract labels (TD = Healthy, FEP = Unhealthy)"
echo "  • Process each file with IMPROVED simplicial complex analysis"
echo "  • Use adaptive local thresholding with global minimums as safety nets"
echo "  • Validate structure quality before export"
echo "  • Export TopoNetX-ready artifacts to export_tnx_windows_improved/"
echo ""

echo "Starting IMPROVED EEG data processing..."
python export_windows_classification_lastcase.py

if [ $? -eq 0 ]; then
    echo "✓ Step 1 completed successfully!"
    echo "  Data exported to export_tnx_windows_improved/"
else
    echo "✗ Step 1 failed with exit code $?"
    echo "Please check the error messages above"
    exit 1
fi

# Check if output directory was created and contains data
if [ ! -d "export_tnx_windows_improved" ] || [ -z "$(ls -A export_tnx_windows_improved 2>/dev/null)" ]; then
    echo "Error: export_tnx_windows_improved directory is empty or was not created"
    exit 1
fi

# Check if manifest was created
if [ ! -f "export_tnx_windows_improved/manifest.json" ]; then
    echo "Error: manifest.json not found in export_tnx_windows_improved/"
    exit 1
fi

echo ""
echo "Data Summary:"
echo "============="
# Extract and display summary from manifest
if command -v jq &> /dev/null; then
    total_subjects=$(jq '.subjects | length' export_tnx_windows_improved/manifest.json)
    total_windows=$(jq '.total_windows' export_tnx_windows_improved/manifest.json)
    echo "  Total subjects: $total_subjects"
    echo "  Total windows: $total_windows"
    
    # Display quality metrics if available
    if jq -e '.quality_metrics' export_tnx_windows_improved/manifest.json > /dev/null 2>&1; then
        avg_edges=$(jq '.quality_metrics.avg_edges_per_window' export_tnx_windows_improved/manifest.json)
        avg_triangles=$(jq '.quality_metrics.avg_triangles_per_window' export_tnx_windows_improved/manifest.json)
        echo "  Average edges per window: $avg_edges"
        echo "  Average triangles per window: $avg_triangles"
    fi
else
    echo "  (Install 'jq' for detailed statistics)"
fi

echo ""

# Step 2: Train SCCNN classification model
echo "Step 2: Training SCCNN Classification Model"
echo "==========================================="
echo "This step will:"
echo "  • Split data into train/test sets by subject"
echo "  • Train a Simplicial Complex Convolutional Neural Network (SCCNN)"
echo "  • Evaluate model performance"
echo "  • Generate classification metrics and visualizations"
echo ""

echo "Starting SCCNN model training..."
python train_tmx_classification_sccnn.py \
    --data export_tnx_windows_improved \
    --epochs 50 \
    --lr 0.001 \
    --hidden_edge 64 \
    --gru_hidden 128 \
    --test_split 0.2 \
    --output trained_sccnn_model

if [ $? -eq 0 ]; then
    echo "✓ Step 2 completed successfully!"
    echo "  Model saved to trained_sccnn_model/"
else
    echo "✗ Step 2 failed with exit code $?"
    echo "Please check the error messages above"
    exit 1
fi

# Check if training results were generated
if [ ! -f "trained_sccnn_model/results.json" ]; then
    echo "Error: Training results not found in trained_sccnn_model/"
    exit 1
fi

echo ""

# Step 3: Generate final summary and copy results
echo "Step 3: Finalizing Results"
echo "=========================="

# Copy results to pipeline_results_improved
echo "Copying results to pipeline_results_improved/..."
cp -r trained_sccnn_model/* pipeline_results_improved/
cp export_tnx_windows_improved/manifest.json pipeline_results_improved/

# Generate final summary
echo "Generating final summary..."
cat > pipeline_results_improved/PIPELINE_SUMMARY.txt << EOF
Imperial Project - EEG Classification Pipeline Results (IMPROVED VERSION)
=======================================================================

Pipeline completed on: $(date)

IMPROVEMENTS IMPLEMENTED:
- Adaptive local thresholding with global minimums as safety nets
- Structure quality validation (minimum 5 edges, 2 triangles)
- Less aggressive global thresholds (10th/25th percentiles vs min/97.5th)
- Quality monitoring metrics for data generation

Data Processing Summary:
- Input directory: ../eeg/
- Output directory: export_tnx_windows_improved/
- Total subjects processed: $(jq '.subjects | length' export_tnx_windows_improved/manifest.json 2>/dev/null || echo "N/A")
- Total windows generated: $(jq '.total_windows' export_tnx_windows_improved/manifest.json 2>/dev/null || echo "N/A")

Quality Metrics:
- Average edges per window: $(jq '.quality_metrics.avg_edges_per_window' export_tnx_windows_improved/manifest.json 2>/dev/null || echo "N/A")
- Average triangles per window: $(jq '.quality_metrics.avg_triangles_per_window' export_tnx_windows_improved/manifest.json 2>/dev/null || echo "N/A")

Training Summary:
- Model architecture: Simplicial Complex Convolutional Neural Network (SCCNN)
- Training epochs: 50
- Learning rate: 0.001
- Test split ratio: 0.2
- Best validation accuracy: $(jq '.best_val_accuracy' trained_sccnn_model/results.json 2>/dev/null || echo "N/A")
- Final test accuracy: $(jq '.test_accuracy' trained_sccnn_model/results.json 2>/dev/null || echo "N/A")

Output Files:
- Trained model: trained_sccnn_model/best_model.pth
- Training results: trained_sccnn_model/results.json
- Training plots: trained_sccnn_model/training_history.png
- Confusion matrix: trained_sccnn_model/confusion_matrix.png
- Pipeline results: pipeline_results_improved/

Label Mapping:
- Label 0: Healthy patients (TD files)
- Label 1: Unhealthy patients (FEP files)

EOF

echo "✓ Final summary generated"
echo ""

# Display final results
echo "Pipeline Results Summary:"
echo "========================"
if [ -f "trained_sccnn_model/results.json" ]; then
    if command -v jq &> /dev/null; then
        echo "  Test Accuracy: $(jq -r '.test_accuracy' trained_sccnn_model/results.json)"
        echo "  Best Validation Accuracy: $(jq -r '.best_val_accuracy' trained_sccnn_model/results.json)"
        echo "  Test Loss: $(jq -r '.test_loss' trained_sccnn_model/results.json)"
    else
        echo "  Results saved to trained_sccnn_model/results.json"
        echo "  Install 'jq' to view detailed results"
    fi
fi

echo ""
echo "Output Files:"
echo "============="
echo "  • export_tnx_windows_improved/ - Processed EEG data windows (IMPROVED)"
echo "  • trained_sccnn_model/ - Trained SCCNN model and training results"
echo "  • pipeline_results_improved/ - Complete pipeline results"
echo "  • pipeline_results_improved/PIPELINE_SUMMARY.txt - Detailed summary"

echo ""
echo "=================================================="
echo " Pipeline completed successfully "
echo "=================================================="
echo ""
echo "Next steps:"
echo "  1. Review results in pipeline_results_improved/"
echo "  2. Check training plots and confusion matrix"
echo "  3. Analyze model performance metrics"
echo "  4. Use trained model for new predictions"
echo ""
echo "For detailed analysis, check:"
echo "  • trained_sccnn_model/results.json - Complete results"
echo "  • trained_sccnn_model/training_history.png - Training curves"
echo "  • trained_sccnn_model/confusion_matrix.png - Classification performance"
echo ""