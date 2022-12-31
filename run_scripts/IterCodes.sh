max=10
for i in `seq 1 $max`
do
    python $1 --model_name="run_$i.pth"
done
