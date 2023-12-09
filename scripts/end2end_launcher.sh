#!/bin/bash

# Define default values for the optional arguments
SHOW_COMMAND_OUTPUT_DEFAULT=false
DEBUG_DEFAULT=false
DEBUG_K_DEFAULT=null
LOGGER_DEFAULT="wandb"
N_INDEPENDENT_SAMPLES_DEFAULT=1
BUCKET_DEBUG_K_DEFAULT=null
ONLINE_JUDGE_VERDICTS_COLLECTION_DELAY_DEFAULT=60 # seconds
ONLINE_JUDGE_SINGLE_THREADED_DEFAULT=false
COMPUTE_PERFORMANCE_ON_HIDDEN_TEST_CASES_DEFAULT=true
COMPUTE_PERFORMANCE_ON_PUBLIC_TEST_CASES_DEFAULT=false
ONLINE_EVAL_RETRIES_DEFAULT=3
BUCKETINGS_TO_CONSIDER_DEFAULT="" # empty string means no bucketing
DRY_RUN_DEFAULT=false
SHOW_INFERENCE_CONFIG=false
INFERENCE_OVERRIDES_DEFAULT=""
INFERENCE_ONLY=false
EVALUATION_OVERRIDES_DEFAULT=""
COMPUTE_OVERALL_METRICS=true
LIBRARY_VERSION=false

_cmd_online_judge_single_threaded=""

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --exp-prefix)
      EXP_PREFIX="$2"
      shift 2
      ;;
    --config-inference)
      CONFIG_INFERENCE="$2"
      shift 2
      ;;
    --configs-evaluation)
      CONFIGS_EVALUATION="$2"
      shift 2
      ;;
    --config-metrics-calculation)
      CONFIG_METRICS_CALCULATION="$2"
      shift 2
      ;;
    --bucketings-to-consider)
      BUCKETINGS_TO_CONSIDER="$2"
      shift 2
      ;;
    # Optional arguments
    --show-command-output)
      SHOW_COMMAND_OUTPUT=true
      shift 1
      ;;
    --debug)
      DEBUG=true
      shift 1
      ;;
    --debug-k)
      DEBUG_K="$2"
      shift 2
      ;;
    --logger)
      LOGGER="$2"
      shift 2
      ;;
    --n-independent-samples)
      N_INDEPENDENT_SAMPLES="$2"
      shift 2
      ;;
    --bucket-debug-k)
      BUCKET_DEBUG_K="$2"
      shift 2
      ;;
    --online-judge-verdicts-collection-delay)
      ONLINE_JUDGE_VERDICTS_COLLECTION_DELAY="$2"
      shift 2
      ;;
    --compute-performance-on-hidden-test-cases)
      COMPUTE_PERFORMANCE_ON_HIDDEN_TEST_CASES="$2"
      shift 2
      ;;
    --compute-performance-on-public-test-cases)
      COMPUTE_PERFORMANCE_ON_PUBLIC_TEST_CASES="$2"
      shift 2
      ;;
    --online-judge-single-threaded)
      ONLINE_JUDGE_SINGLE_THREADED=true
      _cmd_online_judge_single_threaded="--online-judge-single-threaded"
      shift 1
      ;;
    --online-eval-retries)
      ONLINE_EVAL_RETRIES="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=true
      shift 1
      ;;
    --show-inference-config)
      SHOW_INFERENCE_CONFIG=true
      shift 1
      ;;
    --inference-overrides)
      INFERENCE_OVERRIDES="$2"
      shift 2
      ;;
    --evaluation-overrides)
      EVALUATION_OVERRIDES="$2"
      shift 2
      ;;
    --no-compute-overall-metrics)
      COMPUTE_OVERALL_METRICS=false
      shift 1
      ;;
    --inference-only)
      INFERENCE_ONLY=true
      shift 1
      ;;
    *)
      echo "Invalid argument: $1" >&2
      exit 1
      ;;
  esac
done

# Check if mandatory arguments have been set
if [[ -z $EXP_PREFIX || -z $CONFIG_INFERENCE || -z $CONFIGS_EVALUATION || -z $CONFIG_METRICS_CALCULATION ]]; then
  echo "Error: Mandatory arguments --exp_prefix, --config-inference, --configs-evaluation, and --config-metrics-calculation must be provided" >&2
  exit 1
fi

# Set default values for optional arguments if not provided
SHOW_COMMAND_OUTPUT="${SHOW_COMMAND_OUTPUT:-$SHOW_COMMAND_OUTPUT_DEFAULT}"
DEBUG="${DEBUG:-$DEBUG_DEFAULT}"
DEBUG_K="${DEBUG_K:-$DEBUG_K_DEFAULT}"
LOGGER="${LOGGER:-$LOGGER_DEFAULT}"
N_INDEPENDENT_SAMPLES="${N_INDEPENDENT_SAMPLES:-$N_INDEPENDENT_SAMPLES_DEFAULT}"
BUCKET_DEBUG_K="${BUCKET_DEBUG_K:-$BUCKET_DEBUG_K_DEFAULT}"
ONLINE_JUDGE_VERDICTS_COLLECTION_DELAY="${ONLINE_JUDGE_VERDICTS_COLLECTION_DELAY:-$ONLINE_JUDGE_VERDICTS_COLLECTION_DELAY_DEFAULT}"
ONLINE_JUDGE_SINGLE_THREADED="${ONLINE_JUDGE_SINGLE_THREADED:-$ONLINE_JUDGE_SINGLE_THREADED_DEFAULT}"
COMPUTE_PERFORMANCE_ON_HIDDEN_TEST_CASES="${COMPUTE_PERFORMANCE_ON_HIDDEN_TEST_CASES:-$COMPUTE_PERFORMANCE_ON_HIDDEN_TEST_CASES_DEFAULT}"
COMPUTE_PERFORMANCE_ON_PUBLIC_TEST_CASES="${COMPUTE_PERFORMANCE_ON_PUBLIC_TEST_CASES:-$COMPUTE_PERFORMANCE_ON_PUBLIC_TEST_CASES_DEFAULT}"
ONLINE_EVAL_RETRIES="${ONLINE_EVAL_RETRIES:-$ONLINE_EVAL_RETRIES_DEFAULT}"
BUCKETINGS_TO_CONSIDER="${BUCKETINGS_TO_CONSIDER:-$BUCKETINGS_TO_CONSIDER_DEFAULT}"
DRY_RUN="${DRY_RUN:-$DRY_RUN_DEFAULT}"
SHOW_INFERENCE_CONFIG="${SHOW_INFERENCE_CONFIG:-$SHOW_INFERENCE_CONFIG_DEFAULT}"
INFERENCE_OVERRIDES="${INFERENCE_OVERRIDES:-$INFERENCE_OVERRIDES_DEFAULT}"
EVALUATION_OVERRIDES="${EVALUATION_OVERRIDES:-$EVALUATION_OVERRIDES_DEFAULT}"
INFERENCE_ONLY="${INFERENCE_ONLY:-$INFERENCE_ONLY_DEFAULT}"
COMPUTE_OVERALL_METRICS="${COMPUTE_OVERALL_METRICS:-$COMPUTE_OVERALL_METRICS_DEFAULT}"

# Print arguments
echo "Arguments:"
echo "  --exp-prefix: \"$EXP_PREFIX\""
echo "  --config-inference: \"$CONFIG_INFERENCE\""
echo "  --configs-evaluation: \"$CONFIGS_EVALUATION\""
echo "  --config-metrics-calculation: \"$CONFIG_METRICS_CALCULATION\""
echo "  --bucketings-to-consider: \"$BUCKETINGS_TO_CONSIDER\""
echo "  --show-command-output: \"$SHOW_COMMAND_OUTPUT\""
echo "  --debug: \"$DEBUG\""
echo "  --debug-k: \"$DEBUG_K\""
echo "  --logger: \"$LOGGER\""
echo "  --n-independent-samples: \"$N_INDEPENDENT_SAMPLES\""
echo "  --bucket-debug-k: \"$BUCKET_DEBUG_K\""
echo "  --online-judge-verdicts-collection-delay: \"$ONLINE_JUDGE_VERDICTS_COLLECTION_DELAY\""
echo "  --online-judge-single-threaded: \"$ONLINE_JUDGE_SINGLE_THREADED\""
echo "  --compute-performance-on-hidden-test-cases: \"$COMPUTE_PERFORMANCE_ON_HIDDEN_TEST_CASES\""
echo "  --compute-performance-on-public-test-cases: \"$COMPUTE_PERFORMANCE_ON_PUBLIC_TEST_CASES\""
echo "  --online-eval-retries: \"$ONLINE_EVAL_RETRIES\""
echo "  --dry-run: \"$DRY_RUN\""
echo "  --show-inference-config: \"$SHOW_INFERENCE_CONFIG\""
echo "  --inference-overrides: \"$INFERENCE_OVERRIDES\""
echo "  --inference-only: \"$INFERENCE_ONLY\""
echo "  --evaluation-overrides: \"$EVALUATION_OVERRIDES\""
echo "  --compute-overall-metrics: \"$COMPUTE_OVERALL_METRICS\""
echo ""

export PYTHONPATH="."

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
##### Inference

# Create a temporary file to store the output
temp_file=$(mktemp)

inference_script="run_inference.py"

echo "[Inference] Run inference for config '$CONFIG_INFERENCE'"
# Run the command, display the output on the command line, and save the output to the temporary file
if [ "$SHOW_COMMAND_OUTPUT" = "true" ]; then
    echo "Executing: python $inference_script +experiment=$CONFIG_INFERENCE datamodule.debug=$DEBUG datamodule.debug_k=$DEBUG_K datamodule.bucket_debug_k=$BUCKET_DEBUG_K model.n_independent_samples=$N_INDEPENDENT_SAMPLES logger=$LOGGER prefix=$EXP_PREFIX $INFERENCE_OVERRIDES"
    echo
    if [ "$DRY_RUN" = "true" ]; then
      if [ "$SHOW_INFERENCE_CONFIG" = "true" ]; then
        echo "The inference will run with the following configuration:"
        echo
        python $inference_script +experiment=$CONFIG_INFERENCE \
                              datamodule.debug=$DEBUG datamodule.debug_k=$DEBUG_K datamodule.bucket_debug_k=$BUCKET_DEBUG_K \
                              model.n_independent_samples=$N_INDEPENDENT_SAMPLES \
                              logger=$LOGGER prefix=$EXP_PREFIX \
                              $INFERENCE_OVERRIDES --cfg job --resolve
      fi
      echo "Evaluation call: bash scripts/eval_and_metrics_launcher.sh --wandb-run-path ??? --configs-evaluation \"$CONFIGS_EVALUATION\" --config-metrics-calculation \"$CONFIG_METRICS_CALCULATION\" --bucketings-to-consider \"$BUCKETINGS_TO_CONSIDER\" --logger \"$LOGGER\" --online-judge-verdicts-collection-delay \"$ONLINE_JUDGE_VERDICTS_COLLECTION_DELAY\" $_cmd_online_judge_single_threaded --compute-performance-on-hidden-test-cases \"$COMPUTE_PERFORMANCE_ON_HIDDEN_TEST_CASES\" --compute-performance-on-public-test-cases \"$COMPUTE_PERFORMANCE_ON_PUBLIC_TEST_CASES\" --online-eval-retries \"$ONLINE_EVAL_RETRIES\" $EVALUATION_OVERRIDES"
      exit 0
    fi
    python $inference_script +experiment=$CONFIG_INFERENCE \
                            datamodule.debug=$DEBUG datamodule.debug_k=$DEBUG_K datamodule.bucket_debug_k=$BUCKET_DEBUG_K \
                            model.n_independent_samples=$N_INDEPENDENT_SAMPLES \
                            $INFERENCE_OVERRIDES logger=$LOGGER prefix=$EXP_PREFIX | tee "$temp_file"
    # Capture the return code (exit status) of the command
    return_code=${PIPESTATUS[0]}
else
    echo "Executing: python $inference_script +experiment=$CONFIG_INFERENCE datamodule.debug=$DEBUG datamodule.debug_k=$DEBUG_K datamodule.bucket_debug_k=$BUCKET_DEBUG_K model.n_independent_samples=$N_INDEPENDENT_SAMPLES logger=$LOGGER prefix=$EXP_PREFIX"
    if [ "$DRY_RUN" = "true" ]; then
      echo "Evaluation call: bash scripts/eval_and_metrics_launcher.sh --wandb-run-path ??? --configs-evaluation \"$CONFIGS_EVALUATION\" --config-metrics-calculation \"$CONFIG_METRICS_CALCULATION\" --bucketings-to-consider \"$BUCKETINGS_TO_CONSIDER\" --logger \"$LOGGER\" --online-judge-verdicts-collection-delay \"$ONLINE_JUDGE_VERDICTS_COLLECTION_DELAY\" $_cmd_online_judge_single_threaded --compute-performance-on-hidden-test-cases \"$COMPUTE_PERFORMANCE_ON_HIDDEN_TEST_CASES\" --compute-performance-on-public-test-cases \"$COMPUTE_PERFORMANCE_ON_PUBLIC_TEST_CASES\" --online-eval-retries \"$ONLINE_EVAL_RETRIES\" $EVALUATION_OVERRIDES"
      exit 0
    fi
    python $inference_script +experiment=$CONFIG_INFERENCE \
                            datamodule.debug=$DEBUG datamodule.debug_k=$DEBUG_K datamodule.bucket_debug_k=$BUCKET_DEBUG_K \
                            model.n_independent_samples=$N_INDEPENDENT_SAMPLES \
                            logger=$LOGGER prefix=$EXP_PREFIX > "$temp_file"
    # Capture the return code (exit status) of the command
    return_code=$?
fi

# Check if the command execution was successful
if [ $return_code -ne 0 ]; then
    echo
    echo "[Inference] An error with return code '$return_code' occurred."
    echo
    exit $return_code
fi

# Remove the temporary file
wandb_path_inference=$(cat "$temp_file" | grep "WandB run path" | awk -F'\`' '{for (i=2; i<=NF; i+=2) print $i}')
output_directory_inference=$(cat "$temp_file" | grep "Output directory:" | head -n 1 | awk -F'\`' '{for (i=2; i<=NF; i+=2) print $i}')
rm "$temp_file"
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
echo
echo "[Inference] WandB path: $wandb_path_inference"
echo "[Inference] Output directory: $output_directory_inference"
echo

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
##### Evaluation and Metrics Calculation
echo
echo "[Evaluation] Continuing with evaluation and metrics calculation"
echo "[Evaluation] In case of failure, you can restart the part that follows by executing the following command:"
echo
echo "bash scripts/eval_and_metrics_launcher.sh --wandb-run-path $wandb_path_inference --configs-evaluation \"$CONFIGS_EVALUATION\" --config-metrics-calculation \"$CONFIG_METRICS_CALCULATION\" --bucketings-to-consider \"$BUCKETINGS_TO_CONSIDER\" --logger \"$LOGGER\" --online-judge-verdicts-collection-delay \"$ONLINE_JUDGE_VERDICTS_COLLECTION_DELAY\" $_cmd_online_judge_single_threaded --compute-performance-on-hidden-test-cases \"$COMPUTE_PERFORMANCE_ON_HIDDEN_TEST_CASES\" --compute-performance-on-public-test-cases \"$COMPUTE_PERFORMANCE_ON_PUBLIC_TEST_CASES\" --online-eval-retries \"$ONLINE_EVAL_RETRIES\" $EVALUATION_OVERRIDES"
echo
echo

if [ "$INFERENCE_ONLY" = "true" ]; then
  echo "[Evaluation] Skipping evaluation and metrics calculation because --inference-only is set to true"
  exit 0
fi

# Split the strings into arrays using the space separator
IFS=' ' read -r -a CONFIGS_EVALUATION <<< "$CONFIGS_EVALUATION"

metrics_config=$(echo "$CONFIG_METRICS_CALCULATION" | tr ' ' ',')

for evaluation_config_name in "${CONFIGS_EVALUATION[@]}"
do
    # ~~~ Run the evaluation ~~~
    # Get the code_evaluator_id & collect the verdicts for the online judge
    if [[ $evaluation_config_name == *"online_judge"* ]]
    then
      code_evaluator_id="online_judge"

      for i in $(seq 1 $ONLINE_EVAL_RETRIES); do
        echo "Executing: python run_evaluation.py +experiment=$evaluation_config_name wandb_run_path=$wandb_path_inference logger=$LOGGER debug=$ONLINE_JUDGE_SINGLE_THREADED_DEFAULT $EVALUATION_OVERRIDES"
        python run_evaluation.py +experiment=$evaluation_config_name \
                         wandb_run_path=$wandb_path_inference \
                         logger=$LOGGER \
                         debug=$ONLINE_JUDGE_SINGLE_THREADED_DEFAULT $EVALUATION_OVERRIDES
        ONLINE_JUDGE_SINGLE_THREADED_DEFAULT=true
      done

      # Check if the command execution was successful
      return_code=$?
      if [ $return_code -ne 0 ]; then
          echo
          echo "[Online evaluation -- submission] An error with return code '$return_code' occurred."
          echo
          exit $return_code
      fi

      # if we are running the codeforces online evaluation we need to collect the verdicts
      if [[ $evaluation_config_name == *"codeforces"* ]]
      then
        # Delay for the online judge to actually perform the evaluation
        sleep $ONLINE_JUDGE_VERDICTS_COLLECTION_DELAY

        # Collect verdicts for the online judge
        echo "Collecting the verdicts for the online judge..."
        echo "Executing: python run_evaluation.py +experiment=$evaluation_config_name wandb_run_path=$wandb_path_inference logger=$LOGGER debug=$ONLINE_JUDGE_SINGLE_THREADED_DEFAULT $EVALUATION_OVERRIDES"
        python run_evaluation.py +experiment=$evaluation_config_name \
                                 wandb_run_path=$wandb_path_inference \
                                 logger=$LOGGER \
                                 debug=$ONLINE_JUDGE_SINGLE_THREADED_DEFAULT $EVALUATION_OVERRIDES

        # Check if the command execution was successful
        return_code=$?
        if [ $return_code -ne 0 ]; then
            echo
            echo "[Online evaluation -- verdict collection] An error with return code '$return_code' occurred."
            echo
            exit $return_code
        fi
      fi
    else
        code_evaluator_id="local_evaluator"

        echo "Executing: python run_evaluation.py +experiment=$evaluation_config_name wandb_run_path=$wandb_path_inference logger=$LOGGER $EVALUATION_OVERRIDES"
        python run_evaluation.py +experiment=$evaluation_config_name \
                                 wandb_run_path=$wandb_path_inference \
                                 logger=$LOGGER $EVALUATION_OVERRIDES

      # Check if the command execution was successful
      return_code=$?
      if [ $return_code -ne 0 ]; then
          echo
          echo "[Local evaluation] An error with return code '$return_code' occurred."
          echo
          exit $return_code
      fi

    fi

    # ~~~ Run the metrics calculation ~~~
#    echo "Sleeping for 30 seconds before metrics calculation..."
#    sleep 30

    # Hidden test cases
    if [[ "$COMPUTE_PERFORMANCE_ON_HIDDEN_TEST_CASES" = "true" && "$COMPUTE_OVERALL_METRICS" = "true" ]]; then
        echo
        echo "[Hidden] Computing overall performance: \"[$metrics_config,_bootstrap]\""
        echo
        echo "Executing: python run_metrics_calculation.py +experiment/metrics_calculation=[$metrics_config,_bootstrap] code_evaluator_id=$code_evaluator_id wandb_run_path=$wandb_path_inference logger=$LOGGER"
        python run_metrics_calculation.py +experiment/metrics_calculation="[$metrics_config,_bootstrap]" \
                                          code_evaluator_id=$code_evaluator_id \
                                          replace_evaluation_output=False \
                                          wandb_run_path=$wandb_path_inference \
                                          logger=$LOGGER

        return_code=$?
        if [ $return_code -ne 0 ]; then
            echo
            echo "[Metrics -- overall hidden] An error with return code '$return_code' occurred."
            echo
            exit $return_code
        fi
    fi

    # Public test cases
    if [[ "$COMPUTE_PERFORMANCE_ON_PUBLIC_TEST_CASES" = "true" && "$COMPUTE_OVERALL_METRICS" = "true" ]]; then
        echo
        echo "[Public] Computing overall performance: \"[$metrics_config,_bootstrap,_public_test_cases]\""
        echo
        echo "Executing: python run_metrics_calculation.py +experiment/metrics_calculation=[$metrics_config,_bootstrap,_public_test_cases] code_evaluator_id=$code_evaluator_id wandb_run_path=$wandb_path_inference logger=$LOGGER"
        python run_metrics_calculation.py +experiment/metrics_calculation=[$metrics_config,_bootstrap,_public_test_cases] \
                                          code_evaluator_id=$code_evaluator_id \
                                          replace_evaluation_output=False \
                                          wandb_run_path=$wandb_path_inference \
                                          logger=$LOGGER

        return_code=$?
        if [ $return_code -ne 0 ]; then
            echo
            echo "[Metrics -- overall public] An error with return code '$return_code' occurred."
            echo
            exit $return_code
        fi
    fi


    for BUCKETING_CONFIG in $BUCKETINGS_TO_CONSIDER; do
      echo
      echo "[Hidden] Computing metrics for bucketing config $BUCKETING_CONFIG: \"[$metrics_config,_bootstrap,$BUCKETING_CONFIG\"]"
      echo
      # Hidden test cases
      if [ "$COMPUTE_PERFORMANCE_ON_HIDDEN_TEST_CASES" = "true" ]; then
        echo "Executing: python run_metrics_calculation.py +experiment/metrics_calculation=[$metrics_config,_bootstrap,$BUCKETING_CONFIG] code_evaluator_id=$code_evaluator_id wandb_run_path=$wandb_path_inference logger=$LOGGER"
        python run_metrics_calculation.py +experiment/metrics_calculation=[$metrics_config,_bootstrap,$BUCKETING_CONFIG] \
                                          code_evaluator_id=$code_evaluator_id \
                                          replace_evaluation_output=False \
                                          wandb_run_path=$wandb_path_inference \
                                          logger=$LOGGER

        return_code=$?
        if [ $return_code -ne 0 ]; then
            echo
            echo "[Metrics -- $BUCKETING_CONFIG hidden] An error with return code '$return_code' occurred."
            echo
            exit $return_code
        fi
      fi

      # Public test cases
      echo
      echo "[Public] Computing metrics for bucketing config $BUCKETING_CONFIG: \"[$metrics_config,_bootstrap,$BUCKETING_CONFIG\"]"
      echo
      if [ "$COMPUTE_PERFORMANCE_ON_PUBLIC_TEST_CASES" = "true" ]; then
        echo "Executing: python run_metrics_calculation.py +experiment/metrics_calculation=[$metrics_config,_bootstrap,_public_test_cases,$BUCKETING_CONFIG] code_evaluator_id=$code_evaluator_id wandb_run_path=$wandb_path_inference logger=$LOGGER"
        python run_metrics_calculation.py +experiment/metrics_calculation=[$metrics_config,_bootstrap,_public_test_cases,$BUCKETING_CONFIG] \
                                          code_evaluator_id=$code_evaluator_id \
                                          replace_evaluation_output=False \
                                          wandb_run_path=$wandb_path_inference \
                                          logger=$LOGGER

        return_code=$?
        if [ $return_code -ne 0 ]; then
            echo
            echo "[Metrics -- $BUCKETING_CONFIG public] An error with return code '$return_code' occurred."
            echo
            exit $return_code
        fi
      fi

    done

done

echo "The WandB run ID for the inference run is: $wandb_path_inference"
