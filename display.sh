module load python

clear
while true; do
    python scripts/stats.py -r "final_runs" -s
    echo Last refreshed at $(date +"%H:%M:%S")

    if [ $(squeue -u $USER | tail -n +2 | wc -l) -eq 0 ]; then
        break
    fi

    sleep 1800
    clear
done
