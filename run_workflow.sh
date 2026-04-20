#!/bin/bash

# TTL Method Evaluation Workflow Script
# Usage: ./run_workflow.sh [options]

set -e

# Default values
START_DATE="2026-02-07"
END_DATE="2026-02-07"
MAX_REQUESTS=1000000
OUTPUT_PATH="workflow_ttl_methods_eval_$(date +%Y-%m-%d_%H%M%S)_10%.txt"
CONTROLLED_CAPACITY=1000
UNCONTROLLED_CAPACITY=9000
LRU_CAPACITY=10000
SCORE_PERCENTILE=0.05
PREFETCH_RATIO=0.05


# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --start-date)
            START_DATE="$2"
            shift 2
            ;;
        --end-date)
            END_DATE="$2"
            shift 2
            ;;
        --partition-date)
            START_DATE="$2"
            END_DATE="$2"
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
        --lru-capacity)
            LRU_CAPACITY="$2"
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
        --help)
            echo "Usage: ./run_workflow.sh [options]"
            echo ""
            echo "Options:"
            echo "  --start-date DATE                  Start date (default: 2026-02-07)"
            echo "  --end-date DATE                    End date (default: 2026-02-07)"
            echo "  --partition-date DATE              Backward-compatible alias for single-day run"
            echo "  --max-requests NUM                 Max requests to sample (default: 100000000)"
            echo "  --output-path PATH                 Output file path (default: workflow_ttl_methods_eval_TIMESTAMP.txt)"
            echo "  --controlled-capacity NUM          Controlled cache capacity (default: 20000)"
            echo "  --uncontrolled-capacity NUM        Uncontrolled cache capacity (default: 180000)"
            echo "  --lru-capacity NUM                 LRU baseline capacity (default: 200000)"
            echo "  --score-percentile FLOAT           Score percentile (default: 0.9)"
            echo "  --prefetch-ratio FLOAT             Prefetch ratio (default: 0.1)"
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
            echo "  # Run with a date range"
            echo "  ./run_workflow.sh --start-date 2026-02-06 --end-date 2026-02-07 --max-requests 3000"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Prefer the project virtualenv interpreter to avoid conda/site-package conflicts.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="$SCRIPT_DIR/.venv/bin/python"
if [ -x "$VENV_PYTHON" ]; then
    PYTHON_BIN="$VENV_PYTHON"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
else
    PYTHON_BIN="python"
fi

# Build the Python command as an argument array (safer than eval).
PYTHON_CMD=(
    "$PYTHON_BIN" "Cache_System_Workflow/cache_pipeline.py"
    --start-date "$START_DATE"
    --end-date "$END_DATE"
    --max-requests "$MAX_REQUESTS"
    --output-path "$OUTPUT_PATH"
    --controlled-capacity "$CONTROLLED_CAPACITY"
    --uncontrolled-capacity "$UNCONTROLLED_CAPACITY"
    --lru-capacity "$LRU_CAPACITY"
    --score-percentile "$SCORE_PERCENTILE"
    --prefetch-ratio "$PREFETCH_RATIO"
)


# Print configuration
echo "=========================================="
echo "TTL Method Evaluation Workflow"
echo "=========================================="
echo "Start Date:               $START_DATE"
echo "End Date:                 $END_DATE"
echo "Max Requests:             $MAX_REQUESTS"
echo "Output Path:              $OUTPUT_PATH"
echo "Controlled Capacity:      $CONTROLLED_CAPACITY"
echo "Uncontrolled Capacity:    $UNCONTROLLED_CAPACITY"
echo "LRU Capacity:             $LRU_CAPACITY"
echo "Score Percentile:         $SCORE_PERCENTILE"
echo "Prefetch Ratio:           $PREFETCH_RATIO"
echo "Python Interpreter:       $PYTHON_BIN"
echo "=========================================="
echo ""

# Run the workflow
echo "Starting workflow execution..."
"${PYTHON_CMD[@]}"

echo ""
echo "=========================================="
echo "Workflow completed successfully!"
echo "Results saved to: $OUTPUT_PATH"
echo "=========================================="
