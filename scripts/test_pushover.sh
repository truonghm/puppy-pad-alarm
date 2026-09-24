#!/usr/bin/env bash
set -euo pipefail

: "${PUSHOVER_TOKEN:?Set PUSHOVER_TOKEN before running this script.}"
: "${PUSHOVER_USER:?Set PUSHOVER_USER before running this script.}"

curl --silent --show-error --fail-with-body \
  --write-out '\nHTTP %{http_code}\n' \
  --data-urlencode "token=$PUSHOVER_TOKEN" \
  --data-urlencode "user=$PUSHOVER_USER" \
  --data-urlencode 'title=Puppy pad connection test' \
  --data-urlencode 'message=Testing Pushover from Git Bash' \
  --data-urlencode 'priority=2' \
  --data-urlencode 'retry=60' \
  --data-urlencode 'expire=300' \
  https://api.pushover.net/1/messages.json
