#!/bin/bash

# TTL Method Evaluation Workflow Script
# Usage: ./run_workflow.sh [options]

set -e

# Default values
PARTITION_DATE="2026-02-07"
MAX_REQUESTS=10000000
OUTPUT_PATH="workflow_ttl_methods_eval_$(date +%Y-%m-%d_%H%M%S).txt"
CONTROLLED_CAPACITY=100
UNCONTROLLED_CAPACITY=900
SCORE_PERCENTILE=0.7
PREFETCH_RATIO=0.2
ENABLE_BACKGROUND_REFRESH=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --partition-date)
            PARTITION_DATE="$2"
            shift 2
            ;;
        --max-requests)
            MAX_REQUESTS="$2"
            shift 2
            ;;
        --output-path)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        --controlled-capacity)
            CONTROLLED_CAPACITY="$2"
            shift 2
            ;;
        --uncontrolled-capacity)
            UNCONTROLLED_CAPACITY="$2"
            shift 2
            ;;
        --score-percentile)
            SCORE_PERCENTILE="$2"
            shift 2
            ;;
        --prefetch-ratio)
            PREFETCH_RATIO="$2"
            shift 2
            ;;
        --enable-background-refresh)
            ENABLE_BACKGROUND_REFRESH=true
            shift
            ;;
        --help)
            echo "Usage: ./run_workflow.sh [options]"
            echo ""
            echo "Options:"
            echo "  --partition-date DATE              Partition date (default: 2026-02-07)"
            echo "  --max-requests NUM                 Max requests to sample (default: 10000000)"
            echo "  --output-path PATH                 Output file path (default: workflow_ttl_methods_eval_TIMESTAMP.txt)"
            echo "  --controlled-capacity NUM          Controlled cache capacity (default: 100)"
            echo "  --uncontrolled-capacity NUM        Uncontrolled cache capacity (default: 900)"
            echo "  --score-percentile FLOAT           Score percentile (default: 0.7)"
            echo "  --prefetch-ratio FLOAT             Prefetch ratio (default: 0.2)"
            echo "  --enable-background-refresh        Enable background refresh flag"
            echo "  --help                             Show this help message"
            echo ""
            echo "Examples:"
            echo "  # Run with defaults"
            echo "  ./run_workflow.sh"
            echo ""
            echo "  # Run with custom cache sizes"
            echo "  ./run_workflow.sh --controlled-capacity 200 --uncontrolled-capacity 800"
            echo ""
            echo "  # Run with specific partition date and request limit"
            echo "  ./run_workflow.sh --partition-date 2026-02-07 --max-requests 5000 --output-path my_result.txt"
            echo ""
            echo "  # Run with background refresh enabled"
            echo "  ./run_workflow.sh --enable-background-refresh"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Build the Python command
PYTHON_CMD="python Cache_System_Workflow/cache_pipeline.py"
PYTHON_CMD="$PYTHON_CMD --partition-date $PARTITION_DATE"
PYTHON_CMD="$PYTHON_CMD --max-requests $MAX_REQUESTS"
PYTHON_CMD="$PYTHON_CMD --output-path $OUTPUT_PATH"
PYTHON_CMD="$PYTHON_CMD --controlled-capacity $CONTROLLED_CAPACITY"
PYTHON_CMD="$PYTHON_CMD --uncontrolled-capacity $UNCONTROLLED_CAPACITY"
PYTHON_CMD="$PYTHON_CMD --score-percentile $SCORE_PERCENTILE"
PYTHON_CMD="$PYTHON_CMD --prefetch-ratio $PREFETCH_RATIO"

if [ "$ENABLE_BACKGROUND_REFRESH" = true ]; then
    PYTHON_CMD="$PYTHON_CMD --enable-background-refresh"
fi

# Print configuration
echo "=========================================="
echo "TTL Method Evaluation Workflow"
echo "=========================================="
echo "Partition Date:           $PARTITION_DATE"
echo "Max Requests:             $MAX_REQUESTS"
echo "Output Path:              $OUTPUT_PATH"
echo "Controlled Capacity:      $CONTROLLED_CAPACITY"
echo "Uncontrolled Capacity:    $UNCONTROLLED_CAPACITY"
echo "Score Percentile:         $SCORE_PERCENTILE"
echo "Prefetch Ratio:           $PREFETCH_RATIO"
echo "Background Refresh:       $ENABLE_BACKGROUND_REFRESH"
echo "=========================================="
echo ""

# Run the workflow
echo "Starting workflow execution..."
eval $PYTHON_CMD

echo ""
echo "=========================================="
echo "Workflow completed successfully!"
echo "Results saved to: $OUTPUT_PATH"
echo "=========================================="
