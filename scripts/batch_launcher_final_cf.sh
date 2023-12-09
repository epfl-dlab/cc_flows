#!/bin/bash

export PYTHONPATH="."
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Initialize counter variable
successful_logs=$(mktemp)
unsuccessful_logs=$(mktemp)
counter=0

DRY_RUN=$1 # Set to `false` to execute the commands
SHOW_INFERENCE_CONFIG="--show-inference-config" # Set to "" to hide the inference config

ONLINE_EVAL_RETRIES=3

BUCKET_DEBUG_K=68 # CF final run
BUCKET_DEBUG_K="1" # ToDo: Comment out for final run

N_INDEPENDENT_SAMPLES=1
CONFIG_METRICS_CALCULATION="solve_rate" # when running with n_independent_samples >= 3

CF_CONFIGS_EVALUATION="evaluation/codeforces_local_evaluator" # to run local evaluation only
#CF_CONFIGS_EVALUATION="evaluation/codeforces_online_judge" # to run online evaluation only
#CF_CONFIGS_EVALUATION="evaluation/codeforces_local_evaluator evaluation/codeforces_online_judge" # to run both local and online evaluation
CF_BUCKETINGS_TO_CONSIDER="_bucketing_codeforces_before_and_after_cutoff_chatgpt"
#CF_BUCKETINGS_TO_CONSIDER="" # ToDo: Comment out for final run
DEBUG=""

EXP_PREFIX="debug-flows_v1--" # ToDo: Update
LOGGER="wandb_team"
LOGGER="wandb" # ToDo: Comment out for final run

INFERENCE_OVERRIDES="--inference-overrides \"fault_tolerant_mode=True\""
# INFERENCE_OVERRIDES="--inference-overrides \"fault_tolerant_mode=True +debug_config=cf_code_debug_success-vs-code_debug_collab\""

EVALUATION_OVERRIDES=""


cf_cmd_list=(
# Code grounding (ToDo: Use the first run with BUCKET_DEBUG_K=2 to quickly verify that everything runs end-to-end)
#"bash scripts/end2end_launcher.sh --exp-prefix $EXP_PREFIX --config-inference \"inference/gpt4/cf-code\" --configs-evaluation \"$CF_CONFIGS_EVALUATION\" --config-metrics-calculation \"$CONFIG_METRICS_CALCULATION\" --bucketings-to-consider \"$CF_BUCKETINGS_TO_CONSIDER\" --bucket-debug-k \"$BUCKET_DEBUG_K\" --n-independent-samples \"$N_INDEPENDENT_SAMPLES\" --show-command-output --logger \"$LOGGER\" --online-eval-retries \"$ONLINE_EVAL_RETRIES\" $INFERENCE_OVERRIDES $INFERENCE_ONLY $DEBUG $EVALUATION_OVERRIDES"
#"bash scripts/end2end_launcher.sh --exp-prefix $EXP_PREFIX --config-inference \"inference/gpt4/cf-code_reflect\" --configs-evaluation \"$CF_CONFIGS_EVALUATION\" --config-metrics-calculation \"$CONFIG_METRICS_CALCULATION\" --bucketings-to-consider \"$CF_BUCKETINGS_TO_CONSIDER\" --bucket-debug-k \"$BUCKET_DEBUG_K\" --n-independent-samples \"$N_INDEPENDENT_SAMPLES\" --show-command-output --logger \"$LOGGER\" --online-eval-retries \"$ONLINE_EVAL_RETRIES\" $INFERENCE_OVERRIDES $INFERENCE_ONLY $DEBUG $EVALUATION_OVERRIDES"
#"bash scripts/end2end_launcher.sh --exp-prefix $EXP_PREFIX --config-inference \"inference/gpt4/cf-code_collab\" --configs-evaluation \"$CF_CONFIGS_EVALUATION\" --config-metrics-calculation \"$CONFIG_METRICS_CALCULATION\" --bucketings-to-consider \"$CF_BUCKETINGS_TO_CONSIDER\" --bucket-debug-k \"$BUCKET_DEBUG_K\" --n-independent-samples \"$N_INDEPENDENT_SAMPLES\" --show-command-output --logger \"$LOGGER\" --online-eval-retries \"$ONLINE_EVAL_RETRIES\" $INFERENCE_OVERRIDES $INFERENCE_ONLY $DEBUG $EVALUATION_OVERRIDES"
"bash scripts/end2end_launcher.sh --exp-prefix $EXP_PREFIX --config-inference \"inference/gpt4/cf-code_debug\" --configs-evaluation \"$CF_CONFIGS_EVALUATION\" --config-metrics-calculation \"$CONFIG_METRICS_CALCULATION\" --bucketings-to-consider \"$CF_BUCKETINGS_TO_CONSIDER\" --bucket-debug-k \"$BUCKET_DEBUG_K\" --n-independent-samples \"$N_INDEPENDENT_SAMPLES\" --show-command-output --logger \"$LOGGER\" --online-eval-retries \"$ONLINE_EVAL_RETRIES\" $INFERENCE_OVERRIDES $INFERENCE_ONLY $DEBUG $EVALUATION_OVERRIDES"
#"bash scripts/end2end_launcher.sh --exp-prefix $EXP_PREFIX --config-inference \"inference/gpt4/cf-code_debug_collab\" --configs-evaluation \"$CF_CONFIGS_EVALUATION\" --config-metrics-calculation \"$CONFIG_METRICS_CALCULATION\" --bucketings-to-consider \"$CF_BUCKETINGS_TO_CONSIDER\" --bucket-debug-k \"$BUCKET_DEBUG_K\" --n-independent-samples \"$N_INDEPENDENT_SAMPLES\" --show-command-output --logger \"$LOGGER\" --online-eval-retries \"$ONLINE_EVAL_RETRIES\" $INFERENCE_OVERRIDES $INFERENCE_ONLY $DEBUG $EVALUATION_OVERRIDES"

# Plan grounding
#"bash scripts/end2end_launcher.sh --exp-prefix $EXP_PREFIX --config-inference \"inference/gpt4/cf-plan-code\" --configs-evaluation \"$CF_CONFIGS_EVALUATION\" --config-metrics-calculation \"$CONFIG_METRICS_CALCULATION\" --bucketings-to-consider \"$CF_BUCKETINGS_TO_CONSIDER\" --bucket-debug-k \"$BUCKET_DEBUG_K\" --n-independent-samples \"$N_INDEPENDENT_SAMPLES\" --show-command-output --logger \"$LOGGER\" --online-eval-retries \"$ONLINE_EVAL_RETRIES\" $INFERENCE_OVERRIDES $INFERENCE_ONLY $DEBUG $EVALUATION_OVERRIDES"
#"bash scripts/end2end_launcher.sh --exp-prefix $EXP_PREFIX --config-inference \"inference/gpt4/cf-plan_reflect-code\" --configs-evaluation \"$CF_CONFIGS_EVALUATION\" --config-metrics-calculation \"$CONFIG_METRICS_CALCULATION\" --bucketings-to-consider \"$CF_BUCKETINGS_TO_CONSIDER\" --bucket-debug-k \"$BUCKET_DEBUG_K\" --n-independent-samples \"$N_INDEPENDENT_SAMPLES\" --show-command-output --logger \"$LOGGER\" --online-eval-retries \"$ONLINE_EVAL_RETRIES\" $INFERENCE_OVERRIDES $INFERENCE_ONLY $DEBUG $EVALUATION_OVERRIDES"
#"bash scripts/end2end_launcher.sh --exp-prefix $EXP_PREFIX --config-inference \"inference/gpt4/cf-plan_collab-code\" --configs-evaluation \"$CF_CONFIGS_EVALUATION\" --config-metrics-calculation \"$CONFIG_METRICS_CALCULATION\" --bucketings-to-consider \"$CF_BUCKETINGS_TO_CONSIDER\" --bucket-debug-k \"$BUCKET_DEBUG_K\" --n-independent-samples \"$N_INDEPENDENT_SAMPLES\" --show-command-output --logger \"$LOGGER\" --online-eval-retries \"$ONLINE_EVAL_RETRIES\" $INFERENCE_OVERRIDES $INFERENCE_ONLY $DEBUG $EVALUATION_OVERRIDES"

# Simulating human intervention on plan level
#"bash scripts/end2end_launcher.sh --exp-prefix $EXP_PREFIX --config-inference \"inference/gpt4/cf-plan_oracle-code\" --configs-evaluation \"$CF_CONFIGS_EVALUATION\" --config-metrics-calculation \"$CONFIG_METRICS_CALCULATION\" --bucketings-to-consider \"$CF_BUCKETINGS_TO_CONSIDER\" --bucket-debug-k \"$BUCKET_DEBUG_K\" --n-independent-samples \"$N_INDEPENDENT_SAMPLES\" --show-command-output --logger \"$LOGGER\" --online-eval-retries \"$ONLINE_EVAL_RETRIES\" $INFERENCE_OVERRIDES $INFERENCE_ONLY $DEBUG $EVALUATION_OVERRIDES"

# Simulating human intervention on plan level and code grounding
#"bash scripts/end2end_launcher.sh --exp-prefix $EXP_PREFIX --config-inference \"inference/gpt4/cf-plan_oracle-code_debug_collab\" --configs-evaluation \"$CF_CONFIGS_EVALUATION\" --config-metrics-calculation \"$CONFIG_METRICS_CALCULATION\" --bucketings-to-consider \"$CF_BUCKETINGS_TO_CONSIDER\" --bucket-debug-k \"$BUCKET_DEBUG_K\" --n-independent-samples \"$N_INDEPENDENT_SAMPLES\" --show-command-output --logger \"$LOGGER\" --online-eval-retries \"$ONLINE_EVAL_RETRIES\" $INFERENCE_OVERRIDES $INFERENCE_ONLY $DEBUG $EVALUATION_OVERRIDES"
)

command_list=("${cf_cmd_list[@]}")

for cmd in "${command_list[@]}";
do
    counter=$((counter + 1))
    echo "Executing command ${counter}:"
    echo
    echo "$cmd"
    echo
    if [ "$DRY_RUN" = "false" ]; then
      eval $cmd
    else
      eval "$cmd --dry-run $SHOW_INFERENCE_CONFIG"
    fi
    return_code=$?
    if [ $return_code -ne 0 ]; then
        echo
        echo "[Call ${counter}] An error with return code '$return_code' occurred." | tee -a $unsuccessful_logs
        echo "$cmd" | tee -a $unsuccessful_logs
        echo | tee -a $unsuccessful_logs
    else
        echo
        echo "[Call ${counter}] Successfully completed." | tee -a $successful_logs
        echo "$cmd" | tee -a $successful_logs
        echo | tee -a $successful_logs
    fi
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
done

echo
echo "~~~ Successful logs ~~~"
cat $successful_logs
echo

echo "~~~ Unsuccessful logs ~~~"
cat $unsuccessful_logs

rm $successful_logs
rm $unsuccessful_logs
