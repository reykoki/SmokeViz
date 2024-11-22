import json
exp_num=0
decoders = ['DeepLabV3Plus', 'DeepLabV3', 'PSPNet', 'LinkNet', 'PAN', 'MAnet', 'UnetPlusPlus']
decoders = ['PSPNet', 'DeepLabV3Plus', 'LinkNet']
decoders = ['LinkNet']
decoders = ['DeepLabV3Plus', 'LinkNet', 'PSPNet', 'MAnet']
test = True
for decoder in decoders:
    for exp_num in range(1):
        with open('{}/exp{}.json'.format(decoder, exp_num)) as fn:
            hyperparams = json.load(fn)
        job_name = hyperparams['job_name']
        run_script = 'run_hera.script'
        if test:
            job_name = 'test_{}'.format(job_name)
            run_script = 'run_test.script'

        print('sbatch --export=CONFIG_FN=./configs/pseudo/{}/exp{}.json --job-name={} --output=logs/{}.log {}'.format(decoder, exp_num, job_name, job_name, run_script))
        exp_num += 1
