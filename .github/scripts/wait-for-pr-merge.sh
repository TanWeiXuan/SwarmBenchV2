#!/usr/bin/env bash
set -euo pipefail

pr_url=${1:?usage: wait-for-pr-merge.sh PR_URL [TIMEOUT_SECONDS]}
timeout_seconds=${2:-1800}
deadline=$((SECONDS + timeout_seconds))

while (( SECONDS < deadline )); do
  read -r state merge_state < <(
    gh pr view "$pr_url" --json state,mergeStateStatus --jq '[.state, .mergeStateStatus] | @tsv'
  )
  case "$state" in
    MERGED)
      echo "Rating publication PR merged: $pr_url"
      exit 0
      ;;
    CLOSED)
      echo "Rating publication PR closed without merging: $pr_url" >&2
      exit 1
      ;;
  esac
  if [[ "$merge_state" == "DIRTY" ]]; then
    echo "Rating publication PR has merge conflicts: $pr_url" >&2
    gh pr checks "$pr_url" || true
    exit 1
  fi
  sleep 10
done

echo "Timed out waiting for rating publication PR to merge: $pr_url" >&2
gh pr checks "$pr_url" || true
exit 1
