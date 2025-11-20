#!/bin/bash

cd /path/to/DEIMv2

# 检查log文件是否存在
LOG_FILE="outputs/deimv2_xxx/log.txt"

if [ ! -f "$LOG_FILE" ]; then
    echo "Log file $LOG_FILE not found!"
    exit 1
fi

echo "Starting training monitor..."
echo "Log file: $LOG_FILE"
echo "Press Ctrl+C to stop monitoring"
echo ""

# 选择监控方式
echo "Choose monitoring mode:"
echo "1) Terminal monitor (recommended for servers)"
echo "2) Plot monitor (saves plots to file)"
echo ""
read -p "Enter choice (1 or 2): " choice

case $choice in
    1)
        echo "Starting terminal monitor..."
        python tools/monitor/terminal_monitor.py --log-file "$LOG_FILE" --update-interval 5
        ;;
    2)
        echo "Starting plot monitor..."
        python tools/monitor/real_time_monitor.py --log-file "$LOG_FILE" --update-interval 10
        ;;
    *)
        echo "Invalid choice, using terminal monitor..."
        python tools/monitor/terminal_monitor.py --log-file "$LOG_FILE" --update-interval 5
        ;;
esac
