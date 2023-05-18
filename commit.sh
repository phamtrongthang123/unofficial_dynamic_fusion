#!/bin/bash
msg=$1
time=$(date)
echo "Committing changes to git at $time with message: $msg"
git add . && git commit -m "Automatic Commit: $time. $msg" && git push origin master