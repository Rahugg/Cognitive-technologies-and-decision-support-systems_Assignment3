#!/bin/zsh
cd "${0:A:h}"
/usr/bin/make run
status=$?
if [ $status -ne 0 ]; then
  echo
  echo "Startup failed. Read the message above, fix the issue, then click this file again."
  read -k 1 "?Press any key to close this window..."
fi
exit $status
