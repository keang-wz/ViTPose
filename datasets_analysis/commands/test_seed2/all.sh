#!/usr/bin/env bash
# Running this file performs all eight test inferences/evaluations sequentially.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
bash datasets_analysis/commands/test_seed2/tg3k.sh
bash datasets_analysis/commands/test_seed2/acdc.sh
bash datasets_analysis/commands/test_seed2/hc18.sh
bash datasets_analysis/commands/test_seed2/oasis.sh
bash datasets_analysis/commands/test_seed2/synapse_ct.sh
bash datasets_analysis/commands/test_seed2/ski10.sh
bash datasets_analysis/commands/test_seed2/chestxray.sh
bash datasets_analysis/commands/test_seed2/luna16_26.sh
