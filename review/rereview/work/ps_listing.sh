#!/bin/bash
# lstat listing of PS (path, size, mtime) excluding review/rereview; argument: output file under rereview/logs
cd /home/user/work/phase_screen && find . -path ./review/rereview -prune -o -printf '%p %s %T@\n' | sort > "$1"
