
exp_num=0
decoders = ['DeepLabV3Plus', 'DeepLabV3', 'PSPNet', 'LinkNet', 'PAN', 'MAnet', 'UnetPlusPlus']
decoders = ['DeepLabV3Plus']
for decoder in decoders:
    for exp_num in range(3):
        print('sbatch --export=CONFIG_FN={}/exp{}.json --job-name=optimize_{}_exp{} run_hera.script'.format(decoder, exp_num, decoder, exp_num))
        exp_num += 1
