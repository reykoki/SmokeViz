import glob
exp_num=0

decoders = ['segformer', 'pspnet']
encoders = ['mit','resnet18']

e_d_exp = {'encoder':[], 'decoder':[], 'exp_num': []}
e_d_fns = glob.glob('*_*_*.json')
e_d_fns = glob.glob('./training/*_*_*.json')
#e_d_fns = glob.glob('./testing_01/*_*_*.json')
e_d_fns.sort()
for fn in e_d_fns:
    fn = fn.split('/')[-1]
    e = fn.split('_')[1]
    d = fn.split('_')[0]
    exp_num = fn.split('_')[-1].split('.json')[0]
    e_d_exp['encoder'].append(e)
    e_d_exp['decoder'].append(d)
    e_d_exp['exp_num'].append(exp_num)

test = True
test = False
print('dl')
for idx, decoder in enumerate(e_d_exp['decoder']):
    encoder = e_d_exp['encoder'][idx]
    exp_num = e_d_exp['exp_num'][idx]
    run_script = 'run_train.script'
    job_name = f'{decoder}_{encoder}_{exp_num}'
    if test:
        job_name = 'test_{}'.format(job_name)
        run_script = 'run_test.script'
    #print(f'sbatch --export=CONFIG_FN=./configs_thresh/{decoder}_{encoder}_{exp_num}.json --job-name=thresh_{job_name} --output=logs/thresh_{job_name}.log {run_script}')
    print(f'sbatch --export=CONFIG_FN=./configs_thresh/training/{decoder}_{encoder}_{exp_num}.json --job-name=thresh_{job_name} --output=logs/thresh_{job_name}.log {run_script}')
