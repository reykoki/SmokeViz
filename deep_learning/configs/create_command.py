
exp_num=0
while exp_num < 10:
    print('sbatch --export=EXP_NUM={} --output=logs/exp{}.log --job-name=exp{} run_model.script'.format(exp_num,exp_num,exp_num))
    exp_num += 1
