for pid in $(ps -e -o pid --no-headers); 
    do sudo cpulimit --pid $pid --limit 50 --background
done