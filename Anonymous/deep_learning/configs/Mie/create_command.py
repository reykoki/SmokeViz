import json
exp_num=0
decoders = ['DeepLabV3Plus', 'DeepLabV3', 'PSPNet', 'LinkNet', 'PAN', 'MAnet', 'UnetPlusPlus']
decoders = ['PSPNet', 'DeepLabV3Plus', 'LinkNet']
decoders = ['LinkNet']
decoders = ['DeepLabV3Plus']
decoders = ['DeepLabV3Plus', 'PSPNet', 'LinkNet', 'MAnet']
for decoder in decoders:
    for exp_num in range(1):
        with open('{}/exp{}.json'.format(decoder, exp_num)) as fn:
            hyperparams = json.load(fn)
        job_name = hyperparams['job_name']
        print('sbatch --export=CONFIG_FN=./configs/Mie/{}/exp{}.json --job-name={} --output=logs/{}.log run_hera.script'.format(decoder, exp_num, job_name, job_name))
        exp_num += 1
