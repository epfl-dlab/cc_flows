#!/bin/bash

# Define default values for the optional arguments
LOGGER_DEFAULT="wandb"
ONLINE_JUDGE_VERDICTS_COLLECTION_DELAY_DEFAULT=60 # seconds
ONLINE_JUDGE_SINGLE_THREADED_DEFAULT=false
COMPUTE_PERFORMANCE_ON_HIDDEN_TEST_CASES_DEFAULT=true
COMPUTE_PERFORMANCE_ON_PUBLIC_TEST_CASES_DEFAULT=false
ONLINE_EVAL_RETRIES_DEFAULT=3
OVERRIDE_EVALUATION_DEFAULT=false
OVERRIDE_METRICS_DEFAULT=false
BUCKETINGS_TO_CONSIDER_DEFAULT="" # empty string means no bucketing
EVALUATION_OVERRIDES_DEFAULT=""
COMPUTE_OVERALL_METRICS=true

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --wandb-run-path)
      WANDB_RUN_PATH="$2"
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
    # Optional arguments
    --bucketings-to-consider)
      BUCKETINGS_TO_CONSIDER="$2"
      shift 2
      ;;
    --logger)
      LOGGER="$2"
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
      shift 1
      ;;
    --override-evaluation)
      OVERRIDE_EVALUATION=true
      shift 1
      ;;
    --override-metrics)
      OVERRIDE_METRICS=true
      shift 1
      ;;
    --online-eval-retries)
      ONLINE_EVAL_RETRIES="$2"
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
    *)
      echo "Invalid argument: $1" >&2
      exit 1
      ;;
  esac
done

# Check if mandatory arguments have been set
if [[ -z $WANDB_RUN_PATH || -z $CONFIGS_EVALUATION || -z $CONFIG_METRICS_CALCULATION ]]; then
  echo "Error: Mandatory arguments --exp_prefix, --config-inference, --configs-evaluation, and --config-metrics-calculation must be provided" >&2
  exit 1
fi

# Set default values for optional arguments if not provided
LOGGER="${LOGGER:-$LOGGER_DEFAULT}"
ONLINE_JUDGE_VERDICTS_COLLECTION_DELAY="${ONLINE_JUDGE_VERDICTS_COLLECTION_DELAY:-$ONLINE_JUDGE_VERDICTS_COLLECTION_DELAY_DEFAULT}"
ONLINE_JUDGE_SINGLE_THREADED="${ONLINE_JUDGE_SINGLE_THREADED:-$ONLINE_JUDGE_SINGLE_THREADED_DEFAULT}"
COMPUTE_PERFORMANCE_ON_HIDDEN_TEST_CASES="${COMPUTE_PERFORMANCE_ON_HIDDEN_TEST_CASES:-$COMPUTE_PERFORMANCE_ON_HIDDEN_TEST_CASES_DEFAULT}"
COMPUTE_PERFORMANCE_ON_PUBLIC_TEST_CASES="${COMPUTE_PERFORMANCE_ON_PUBLIC_TEST_CASES:-$COMPUTE_PERFORMANCE_ON_PUBLIC_TEST_CASES_DEFAULT}"
ONLINE_EVAL_RETRIES="${ONLINE_EVAL_RETRIES:-$ONLINE_EVAL_RETRIES_DEFAULT}"
OVERRIDE_EVALUATION="${OVERRIDE_EVALUATION:-$OVERRIDE_EVALUATION_DEFAULT}"
OVERRIDE_METRICS="${OVERRIDE_METRICS:-$OVERRIDE_METRICS_DEFAULT}"
BUCKETINGS_TO_CONSIDER="${BUCKETINGS_TO_CONSIDER:-$BUCKETINGS_TO_CONSIDER_DEFAULT}"
EVALUATION_OVERRIDES="${EVALUATION_OVERRIDES:-$EVALUATION_OVERRIDES_DEFAULT}"
COMPUTE_OVERALL_METRICS="${COMPUTE_OVERALL_METRICS:-$COMPUTE_OVERALL_METRICS_DEFAULT}"


# Print arguments
echo "Arguments:"
echo "  --wandb_path: \"$WANDB_RUN_PATH\""
echo "  --configs-evaluation: \"$CONFIGS_EVALUATION\""
echo "  --config-metrics-calculation: \"$CONFIG_METRICS_CALCULATION\""
echo "  --bucketings-to-consider: \"$BUCKETINGS_TO_CONSIDER\""
echo "  --online-judge-verdicts-collection-delay: \"$ONLINE_JUDGE_VERDICTS_COLLECTION_DELAY\""
echo "  --online-judge-single-threaded: \"$ONLINE_JUDGE_SINGLE_THREADED\""
echo "  --compute-performance-on-hidden-test-cases: \"$COMPUTE_PERFORMANCE_ON_HIDDEN_TEST_CASES\""
echo "  --compute-performance-on-public-test-cases: \"$COMPUTE_PERFORMANCE_ON_PUBLIC_TEST_CASES\""
echo "  --online-eval-retries: \"$ONLINE_EVAL_RETRIES\""
echo "  --override-evaluation: \"$OVERRIDE_EVALUATION\""
echo "  --override-metrics: \"$OVERRIDE_METRICS\""
echo "  --evaluation-overrides: \"$EVALUATION_OVERRIDES\""
echo "  --compute-overall-metrics: \"$COMPUTE_OVERALL_METRICS\""
echo ""

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
##### Evaluation and Metrics Calculation

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
        echo "Executing: python run_evaluation.py +experiment=$evaluation_config_name wandb_run_path=$WANDB_RUN_PATH logger=$LOGGER debug=$ONLINE_JUDGE_SINGLE_THREADED_DEFAULT $EVALUATION_OVERRIDES"
        python run_evaluation.py +experiment=$evaluation_config_name \
                         wandb_run_path=$WANDB_RUN_PATH \
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
        echo "Executing: python run_evaluation.py +experiment=$evaluation_config_name wandb_run_path=$WANDB_RUN_PATH logger=$LOGGER debug=$ONLINE_JUDGE_SINGLE_THREADED_DEFAULT $EVALUATION_OVERRIDES"
        python run_evaluation.py +experiment=$evaluation_config_name \
                                 wandb_run_path=$WANDB_RUN_PATH \
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

        echo "Executing: python run_evaluation.py +experiment=$evaluation_config_name wandb_run_path=$WANDB_RUN_PATH logger=$LOGGER $EVALUATION_OVERRIDES"
        python run_evaluation.py +experiment=$evaluation_config_name \
                                 wandb_run_path=$WANDB_RUN_PATH \
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
        echo "Executing: python run_metrics_calculation.py +experiment/metrics_calculation=[$metrics_config,_bootstrap] code_evaluator_id=$code_evaluator_id wandb_run_path=$WANDB_RUN_PATH logger=$LOGGER override=$OVERRIDE_METRICS"
        python run_metrics_calculation.py +experiment/metrics_calculation="[$metrics_config,_bootstrap]" \
                                          code_evaluator_id=$code_evaluator_id \
                                          replace_evaluation_output=False \
                                          wandb_run_path=$WANDB_RUN_PATH \
                                          logger=$LOGGER override=$OVERRIDE_METRICS

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
        echo "Executing: python run_metrics_calculation.py +experiment/metrics_calculation=[$metrics_config,_bootstrap,_public_test_cases] code_evaluator_id=$code_evaluator_id wandb_run_path=$WANDB_RUN_PATH logger=$LOGGER override=$OVERRIDE_METRICS"
        python run_metrics_calculation.py +experiment/metrics_calculation=[$metrics_config,_bootstrap,_public_test_cases] \
                                          code_evaluator_id=$code_evaluator_id \
                                          replace_evaluation_output=False \
                                          wandb_run_path=$WANDB_RUN_PATH \
                                          logger=$LOGGER override=$OVERRIDE_METRICS

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
        echo "Executing: python run_metrics_calculation.py +experiment/metrics_calculation=[$metrics_config,_bootstrap,$BUCKETING_CONFIG] code_evaluator_id=$code_evaluator_id wandb_run_path=$WANDB_RUN_PATH logger=$LOGGER override=$OVERRIDE_METRICS"
        python run_metrics_calculation.py +experiment/metrics_calculation=[$metrics_config,_bootstrap,$BUCKETING_CONFIG] \
                                          code_evaluator_id=$code_evaluator_id \
                                          replace_evaluation_output=False \
                                          wandb_run_path=$WANDB_RUN_PATH \
                                          logger=$LOGGER override=$OVERRIDE_METRICS

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
        echo "Executing: python run_metrics_calculation.py +experiment/metrics_calculation=[$metrics_config,_bootstrap,_public_test_cases,$BUCKETING_CONFIG] code_evaluator_id=$code_evaluator_id wandb_run_path=$WANDB_RUN_PATH logger=$LOGGER override=$OVERRIDE_METRICS"
        python run_metrics_calculation.py +experiment/metrics_calculation=[$metrics_config,_bootstrap,_public_test_cases,$BUCKETING_CONFIG] \
                                          code_evaluator_id=$code_evaluator_id \
                                          replace_evaluation_output=False \
                                          wandb_run_path=$WANDB_RUN_PATH \
                                          logger=$LOGGER override=$OVERRIDE_METRICS

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

echo "The WandB run ID for the inference run is: $WANDB_RUN_PATH"
