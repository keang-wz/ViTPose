#!/usr/bin/env bash
# Running this file performs all eight test inferences/evaluations sequentially.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
bash datasets_analysis/commands/test_seed1/tg3k.sh
bash datasets_analysis/commands/test_seed1/acdc.sh
bash datasets_analysis/commands/test_seed1/hc18.sh
bash datasets_analysis/commands/test_seed1/oasis.sh
bash datasets_analysis/commands/test_seed1/synapse_ct.sh
bash datasets_analysis/commands/test_seed1/ski10.sh
bash datasets_analysis/commands/test_seed1/chestxray.sh
bash datasets_analysis/commands/test_seed1/luna16_26.sh
